/*
 * kube_renderer.c - raylib-based neon editor for Kube
 *
 * This file exists to provide a lightweight native editor that can render and
 * edit packed Kube data directly from the mapped storage buffer exposed by
 * Python.
 *
 * The high-level responsibilities are:
 * - interpret the packed cell buffer passed from Python
 * - build face geometry for shell or slice views
 * - draw the 3D scene plus editor UI
 * - let the user pick cells/faces
 * - push edits back through Python callbacks
 *
 * The native layer is intentionally narrow: Python still owns persistence,
 * metadata, and validation of the underlying `Kube` object.
 */

#include "raylib.h"
#include "rlgl.h"
#include "kube_ui_clay.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #define KUBE_API __declspec(dllexport)
#else
  #define KUBE_API __attribute__((visibility("default")))
#endif

enum { VIEW_SHELL = 0, VIEW_SLICE = 1 };

/* Callback used to write one face back into the Python-owned kube buffer. */
typedef int (*SetFaceFn)(int x, int y, int z, int face_idx, int value);
/* Callback used to flush Python-side changes after a native edit. */
typedef int (*FlushFn)(void);
/* Callback used to execute one Python debug-console command. */
typedef int (*RunCodeFn)(
    const char *code,
    int has_selection,
    int sel_x,
    int sel_y,
    int sel_z,
    int sel_face,
    int has_hover,
    int hover_x,
    int hover_y,
    int hover_z,
    int hover_face,
    char *output,
    int output_cap
);

/* Maps a face index to the axis/boundary used to decide shell visibility. */
typedef struct { int axis; int is_min; } FaceBound;
/* CPU-side geometry buffers built from the visible subset of the kube. */
typedef struct { float *positions; uint8_t *colors; unsigned int *indices; int n_verts; int n_idx; } Geo;
/* GPU-side handles that mirror the uploaded geometry buffers. */
typedef struct { unsigned int vao; unsigned int vbo_pos; unsigned int vbo_col; unsigned int vbo_idx; int n_idx; } GPU;
/*
 * Current view settings.
 * - view_mode: shell or slice
 * - slice_axis/slice_index: active slice settings when in slice mode
 * - lod: sampling factor for large kubes
 * - read_only: blocks edits when true
 * - visible_faces: shell-mode face visibility derived from camera position
 */
typedef struct { int view_mode; int slice_axis; int slice_index; int lod; int read_only; int visible_faces[6]; } ViewConfig;
/*
 * Current editor selection and HUD feedback state.
 * - has_selection/x/y/z/face: selected target for editing
 * - status/status_frames: short-lived user feedback messages
 */
typedef struct { int has_selection; int x; int y; int z; int face; char status[160]; int status_frames; } EditorState;

/* Per-face cube vertices used for geometry generation and highlighting. */
static const float FACE_VERTS[6][4][3] = {
    {{-0.5f,+0.5f,-0.5f},{+0.5f,+0.5f,-0.5f},{+0.5f,+0.5f,+0.5f},{-0.5f,+0.5f,+0.5f}},
    {{-0.5f,-0.5f,+0.5f},{+0.5f,-0.5f,+0.5f},{+0.5f,-0.5f,-0.5f},{-0.5f,-0.5f,-0.5f}},
    {{-0.5f,-0.5f,+0.5f},{-0.5f,-0.5f,-0.5f},{-0.5f,+0.5f,-0.5f},{-0.5f,+0.5f,+0.5f}},
    {{+0.5f,-0.5f,-0.5f},{+0.5f,-0.5f,+0.5f},{+0.5f,+0.5f,+0.5f},{+0.5f,+0.5f,-0.5f}},
    {{-0.5f,-0.5f,+0.5f},{+0.5f,-0.5f,+0.5f},{+0.5f,+0.5f,+0.5f},{-0.5f,+0.5f,+0.5f}},
    {{+0.5f,-0.5f,-0.5f},{-0.5f,-0.5f,-0.5f},{-0.5f,+0.5f,-0.5f},{+0.5f,+0.5f,-0.5f}}
};
/* Face-to-boundary lookup used to decide which shell faces are visible. */
static const FaceBound FACE_BOUND[6] = {{1,0},{1,1},{0,1},{0,0},{2,0},{2,1}};
static const char *FACE_NAMES[6] = {"Top","Bottom","Left","Right","Front","Back"};
static const Color PALETTE_8[8] = {
    {153,153,153,255},{51,102,230,255},{230,102,77,255},{230,204,26,255},
    {51,204,77,255},{153,51,204,255},{230,128,26,255},{230,26,26,255}
};
static const Color BG = {5,5,5,255};
static const Color PANEL = {10,10,10,244};
static const Color PANEL_ALT = {18,18,18,236};
static const Color CYAN = {228,228,228,255};
static const Color NEON_MAGENTA = {250,250,250,255};
static const Color AMBER = {154,154,154,255};
static const Color TEXT = {244,244,244,255};
static const Color DIM = {112,112,112,255};
static const Color SUCCESS = {232,232,232,255};
static const Color ERROR_C = {182,182,182,255};
static const char *HELP_TEXT =
    "Navigation\n"
    "Right drag orbit | Mouse wheel zoom | R reset camera\n"
    "\n"
    "Views\n"
    "Tab shell or slice | X/Y/Z axis | PgUp/PgDn slice | +/- LOD\n"
    "\n"
    "Selection\n"
    "Left click pick | 1-6 choose face | Backspace clears selection\n"
    "\n"
    "Editing\n"
    "[ and ] adjust face values | Numpad digits set values when states <= 10\n"
    "\n"
    "Debug\n"
    "` or F2 console | G grid | H/F1 help | Esc closes overlays | Ctrl+Q exits editor\n"
    "\n"
    "Console Tips\n"
    "Drag the console by its header bar | Tab autocomplete | Wheel scroll output\n";
static const char *CONSOLE_COMPLETE_PREFIX = "\035COMPLETE\035";
static Font UI_FONT = {0};
static int UI_FONT_LOADED = 0;
static int UI_HOVERED_THIS_FRAME = 0;

/*
 * SECTION_HELPERS
 *
 * Small math, color, font, and packed-value helpers live here so the rest of
 * the renderer can stay focused on interaction and drawing logic.
 */
static int ilog2_int(int value) { int bits = 0; while ((1 << bits) < value) bits++; return bits; }
static int clamp_int(int value, int min_value, int max_value) { if (value < min_value) return min_value; if (value > max_value) return max_value; return value; }
static float absf(float value) { return value < 0.0f ? -value : value; }
static float clampf(float value, float min_value, float max_value) { if (value < min_value) return min_value; if (value > max_value) return max_value; return value; }
static float lerpf(float start, float end, float amount) { return start + (end - start) * amount; }
static float approachf(float current, float target, float amount) { return lerpf(current, target, clampf(amount, 0.0f, 1.0f)); }
static Color color_for_state(int state, int states) { return states <= 8 ? PALETTE_8[state] : ColorFromHSV((float)state / (float)states * 360.0f, 0.85f, 0.95f); }
static void ui_begin_frame(void) { UI_HOVERED_THIS_FRAME = 0; }
static void ui_mark_hot(int hovered) { if (hovered) UI_HOVERED_THIS_FRAME = 1; }
static int ui_is_hot(void) { return UI_HOVERED_THIS_FRAME; }
static void init_ui_font(void) {
    const char *font_paths[] = {
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    };
    UI_FONT = GetFontDefault();
    for (int i = 0; i < (int)(sizeof(font_paths) / sizeof(font_paths[0])); i++) {
        if (FileExists(font_paths[i])) {
            Font loaded = LoadFontEx(font_paths[i], 28, NULL, 0);
            if (loaded.texture.id != 0) {
                UI_FONT = loaded;
                UI_FONT_LOADED = 1;
                break;
            }
        }
    }
    SetTextureFilter(UI_FONT.texture, TEXTURE_FILTER_BILINEAR);
}
static void shutdown_ui_font(void) { if (UI_FONT_LOADED) UnloadFont(UI_FONT); UI_FONT_LOADED = 0; }
static void ui_draw_text(const char *text, float x, float y, int font_size, Color color) { DrawTextEx(UI_FONT, text, (Vector2){x, y}, (float)font_size, 1.0f, color); }
static Vector2 ui_measure_text(const char *text, int font_size) { return MeasureTextEx(UI_FONT, text, (float)font_size, 1.0f); }

static uint64_t read_packed(const void *grid_data, size_t index, int dtype_bytes) {
    switch (dtype_bytes) {
        case 1: return ((const uint8_t *)grid_data)[index];
        case 2: return ((const uint16_t *)grid_data)[index];
        case 4: return ((const uint32_t *)grid_data)[index];
        case 8: return ((const uint64_t *)grid_data)[index];
        default: return 0;
    }
}

static int get_face_value(uint64_t packed, int face_idx, int bits_per_face, int states) {
    return (int)((packed >> (face_idx * bits_per_face)) & (uint64_t)(states - 1));
}

static int sample_coord(int coord, int limit, int lod) { return (coord == 0 || coord == limit) ? 1 : ((lod <= 1) || (coord % lod == 0)); }
static int sample_shell_cell(int x, int y, int z, int W, int H, int D, int face_index, int lod) {
    switch (face_index) {
        case 0: case 1: return sample_coord(x, W - 1, lod) && sample_coord(z, D - 1, lod);
        case 2: case 3: return sample_coord(y, H - 1, lod) && sample_coord(z, D - 1, lod);
        case 4: case 5: return sample_coord(x, W - 1, lod) && sample_coord(y, H - 1, lod);
        default: return 0;
    }
}
static int sample_slice_cell(int x, int y, int z, int W, int H, int D, int slice_axis, int lod) {
    if (slice_axis == 0) return sample_coord(y, H - 1, lod) && sample_coord(z, D - 1, lod);
    if (slice_axis == 1) return sample_coord(x, W - 1, lod) && sample_coord(z, D - 1, lod);
    return sample_coord(x, W - 1, lod) && sample_coord(y, H - 1, lod);
}
static int cell_in_slice(int x, int y, int z, int axis, int index) { return axis == 0 ? x == index : (axis == 1 ? y == index : z == index); }
static int face_from_normal(Vector3 normal) {
    if (absf(normal.y) > 0.9f) return normal.y > 0.0f ? 0 : 1;
    if (absf(normal.x) > 0.9f) return normal.x < 0.0f ? 2 : 3;
    return normal.z > 0.0f ? 4 : 5;
}
static int slice_face_for_camera(const Camera3D *camera, int slice_axis) {
    if (slice_axis == 0) return camera->position.x >= 0.0f ? 3 : 2;
    if (slice_axis == 1) return camera->position.y >= 0.0f ? 0 : 1;
    return camera->position.z >= 0.0f ? 4 : 5;
}
static void set_status(EditorState *editor, const char *message) {
    strncpy(editor->status, message, sizeof(editor->status) - 1);
    editor->status[sizeof(editor->status) - 1] = '\0';
    editor->status_frames = 300;
}

/*
 * SECTION_GEOMETRY
 *
 * These helpers exist to derive the visible subset of the kube, allocate CPU
 * geometry, and upload it into GPU buffers whenever the view state changes.
 */
static void geo_free(Geo *geo) {
    free(geo->positions); free(geo->colors); free(geo->indices);
    geo->positions = NULL; geo->colors = NULL; geo->indices = NULL; geo->n_verts = 0; geo->n_idx = 0;
}

static int count_faces_for_cell(int x, int y, int z, int W, int H, int D, const ViewConfig *view) {
    int count = 0;
    int coords[3] = {x, y, z};
    int limits[3] = {W - 1, H - 1, D - 1};
    if (view->view_mode == VIEW_SLICE) {
        if (!cell_in_slice(x, y, z, view->slice_axis, view->slice_index)) return 0;
        if (!sample_slice_cell(x, y, z, W, H, D, view->slice_axis, view->lod)) return 0;
        return 6;
    }
    for (int face_index = 0; face_index < 6; face_index++) {
        FaceBound bound = FACE_BOUND[face_index];
        int limit = limits[bound.axis];
        int visible = bound.is_min ? (coords[bound.axis] == 0) : (coords[bound.axis] == limit);
        if (visible && view->visible_faces[face_index] && sample_shell_cell(x, y, z, W, H, D, face_index, view->lod)) count++;
    }
    return count;
}

static Geo build_geometry(const void *grid_data, int W, int H, int D, int states, int dtype_bytes, const ViewConfig *view) {
    Geo geo = {0};
    int bits_per_face = ilog2_int(states);
    int face_count = 0;
    for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) for (int z = 0; z < D; z++) face_count += count_faces_for_cell(x, y, z, W, H, D, view);
    geo.n_verts = face_count * 4;
    geo.n_idx = face_count * 6;
    if (face_count == 0) return geo;
    geo.positions = (float *)malloc((size_t)geo.n_verts * 3 * sizeof(float));
    geo.colors = (uint8_t *)malloc((size_t)geo.n_verts * 4 * sizeof(uint8_t));
    geo.indices = (unsigned int *)malloc((size_t)geo.n_idx * sizeof(unsigned int));
    {
        int vertex_index = 0;
        int index_index = 0;
        for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) for (int z = 0; z < D; z++) {
            int coords[3] = {x, y, z};
            int limits[3] = {W - 1, H - 1, D - 1};
            uint64_t packed;
            float cx, cy, cz;
            if (count_faces_for_cell(x, y, z, W, H, D, view) == 0) continue;
            cx = (float)x - (float)W / 2.0f + 0.5f;
            cy = (float)y - (float)H / 2.0f + 0.5f;
            cz = (float)z - (float)D / 2.0f + 0.5f;
            packed = read_packed(grid_data, (size_t)x * (size_t)(H * D) + (size_t)y * (size_t)D + (size_t)z, dtype_bytes);
            for (int face_index = 0; face_index < 6; face_index++) {
                int include_face = 0;
                if (view->view_mode == VIEW_SLICE) include_face = 1;
                else {
                    FaceBound bound = FACE_BOUND[face_index];
                    int limit = limits[bound.axis];
                    int is_boundary = bound.is_min ? (coords[bound.axis] == 0) : (coords[bound.axis] == limit);
                    include_face = is_boundary && view->visible_faces[face_index] && sample_shell_cell(x, y, z, W, H, D, face_index, view->lod);
                }
                if (!include_face) continue;
                {
                    int face_value = get_face_value(packed, face_index, bits_per_face, states);
                    Color color = color_for_state(face_value, states);
                    unsigned int base = (unsigned int)vertex_index;
                    for (int v = 0; v < 4; v++) {
                        geo.positions[vertex_index * 3 + 0] = cx + FACE_VERTS[face_index][v][0];
                        geo.positions[vertex_index * 3 + 1] = cy + FACE_VERTS[face_index][v][1];
                        geo.positions[vertex_index * 3 + 2] = cz + FACE_VERTS[face_index][v][2];
                        geo.colors[vertex_index * 4 + 0] = color.r;
                        geo.colors[vertex_index * 4 + 1] = color.g;
                        geo.colors[vertex_index * 4 + 2] = color.b;
                        geo.colors[vertex_index * 4 + 3] = color.a;
                        vertex_index++;
                    }
                    geo.indices[index_index++] = base; geo.indices[index_index++] = base + 1; geo.indices[index_index++] = base + 2;
                    geo.indices[index_index++] = base; geo.indices[index_index++] = base + 2; geo.indices[index_index++] = base + 3;
                }
            }
        }
    }
    return geo;
}

static GPU upload_geo(const Geo *geo) {
    GPU gpu = {0};
    gpu.n_idx = geo->n_idx;
    gpu.vao = rlLoadVertexArray();
    rlEnableVertexArray(gpu.vao);
    gpu.vbo_pos = rlLoadVertexBuffer(geo->positions, geo->n_verts * 3 * sizeof(float), 0);
    rlSetVertexAttribute(0, 3, RL_FLOAT, 0, 0, 0);
    rlEnableVertexAttribute(0);
    gpu.vbo_col = rlLoadVertexBuffer(geo->colors, geo->n_verts * 4, 0);
    rlSetVertexAttribute(3, 4, RL_UNSIGNED_BYTE, 1, 0, 0);
    rlEnableVertexAttribute(3);
    gpu.vbo_idx = rlLoadVertexBufferElement(geo->indices, geo->n_idx * sizeof(unsigned int), 0);
    rlDisableVertexArray();
    return gpu;
}

static void gpu_free(GPU *gpu) {
    if (gpu->vao != 0) {
        rlUnloadVertexArray(gpu->vao);
        rlUnloadVertexBuffer(gpu->vbo_pos);
        rlUnloadVertexBuffer(gpu->vbo_col);
        rlUnloadVertexBuffer(gpu->vbo_idx);
        gpu->vao = 0; gpu->vbo_pos = 0; gpu->vbo_col = 0; gpu->vbo_idx = 0; gpu->n_idx = 0;
    }
}

static void update_visible_faces(const Camera3D *camera, int visible_faces[6]) {
    for (int index = 0; index < 6; index++) visible_faces[index] = 0;
    visible_faces[camera->position.y >= 0.0f ? 0 : 1] = 1;
    visible_faces[camera->position.x >= 0.0f ? 3 : 2] = 1;
    visible_faces[camera->position.z >= 0.0f ? 4 : 5] = 1;
}

static int visible_faces_changed(const int lhs[6], const int rhs[6]) { for (int i = 0; i < 6; i++) if (lhs[i] != rhs[i]) return 1; return 0; }

/*
 * SECTION_UI
 *
 * This section owns the neon-styled editor chrome, hover/selection effects,
 * mutation helpers, and mouse picking support.
 */
static void draw_panel(Rectangle rect, Color fill, Color outline, Color glow) {
    float time_value = (float)GetTime();
    float pulse = 0.55f + 0.45f * sinf(time_value * 2.2f);
    Rectangle inset = {rect.x + 2.0f, rect.y + 2.0f, rect.width - 4.0f, rect.height - 4.0f};
    Rectangle accent = {rect.x + 10.0f, rect.y + 6.0f, rect.width * (0.18f + 0.12f * pulse), 3.0f};
    DrawRectangleRounded(rect, 0.08f, 8, ColorAlpha(glow, 0.10f + 0.05f * pulse));
    DrawRectangleRounded(inset, 0.08f, 8, fill);
    DrawRectangleRoundedLinesEx(rect, 0.08f, 8, 1.5f, ColorAlpha(outline, 0.72f + 0.18f * pulse));
    DrawRectangleLinesEx((Rectangle){rect.x - 2, rect.y - 2, rect.width + 4, rect.height + 4}, 1.0f, ColorAlpha(glow, 0.20f + 0.08f * pulse));
    DrawRectangleRounded(accent, 0.7f, 6, ColorAlpha(glow, 0.55f));
    DrawCircleV((Vector2){rect.x + 12.0f, rect.y + rect.height - 10.0f}, 2.0f, ColorAlpha(glow, 0.65f));
    DrawCircleV((Vector2){rect.x + rect.width - 12.0f, rect.y + 10.0f}, 2.0f, ColorAlpha(outline, 0.45f));
}

static int button(Rectangle rect, const char *label, Color base, Color outline, Color text_color, int active) {
    Vector2 mouse = GetMousePosition();
    int hovered = CheckCollisionPointRec(mouse, rect);
    int pressed = hovered && IsMouseButtonDown(MOUSE_BUTTON_LEFT);
    float time_value = (float)GetTime();
    float sweep = fmodf(time_value * 110.0f, rect.width + 42.0f) - 42.0f;
    Rectangle sweep_rect = {rect.x + sweep, rect.y + 4.0f, 36.0f, rect.height - 8.0f};
    Color fill = hovered ? ColorAlpha(base, 0.98f) : ColorAlpha(base, active ? 0.92f : 0.72f);
    Vector2 label_size = ui_measure_text(label, 18);
    ui_mark_hot(hovered);
    if (pressed) fill = ColorAlpha(outline, 0.38f);
    draw_panel(rect, fill, outline, outline);
    if (hovered || active) DrawRectangleRounded(sweep_rect, 0.30f, 6, ColorAlpha(text_color, hovered ? 0.11f : 0.08f));
    if (active) DrawRectangleRounded((Rectangle){rect.x + 6.0f, rect.y + rect.height - 5.0f, rect.width - 12.0f, 3.0f}, 0.7f, 6, ColorAlpha(outline, 0.95f));
    if (hovered) DrawCircleV((Vector2){rect.x + rect.width - 12.0f, rect.y + 10.0f}, 2.5f, outline);
    ui_draw_text(label, rect.x + (rect.width - label_size.x) * 0.5f, rect.y + (rect.height - label_size.y) * 0.5f - 1.0f, 18, text_color);
    return hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static void draw_scanlines(int width, int height) {
    int offset = ((int)(GetTime() * 18.0)) % 4;
    for (int y = offset; y < height; y += 4) DrawLine(0, y, width, y, ColorAlpha(TEXT, 0.035f));
}

static void draw_viewport_overlay(Rectangle viewport, int hover_has_cell, float intro) {
    float pulse = 0.5f + 0.5f * sinf((float)GetTime() * 1.8f);
    Color border = hover_has_cell ? ColorAlpha(TEXT, 0.78f) : ColorAlpha(CYAN, 0.38f + 0.12f * pulse);
    float corner = 26.0f + 10.0f * intro;
    DrawRectangleLinesEx(viewport, 1.2f, border);
    DrawLineEx((Vector2){viewport.x + 8.0f, viewport.y + 8.0f}, (Vector2){viewport.x + corner, viewport.y + 8.0f}, 2.0f, border);
    DrawLineEx((Vector2){viewport.x + 8.0f, viewport.y + 8.0f}, (Vector2){viewport.x + 8.0f, viewport.y + corner}, 2.0f, border);
    DrawLineEx((Vector2){viewport.x + viewport.width - 8.0f, viewport.y + 8.0f}, (Vector2){viewport.x + viewport.width - corner, viewport.y + 8.0f}, 2.0f, border);
    DrawLineEx((Vector2){viewport.x + viewport.width - 8.0f, viewport.y + 8.0f}, (Vector2){viewport.x + viewport.width - 8.0f, viewport.y + corner}, 2.0f, border);
    DrawLineEx((Vector2){viewport.x + 8.0f, viewport.y + viewport.height - 8.0f}, (Vector2){viewport.x + corner, viewport.y + viewport.height - 8.0f}, 2.0f, border);
    DrawLineEx((Vector2){viewport.x + 8.0f, viewport.y + viewport.height - 8.0f}, (Vector2){viewport.x + 8.0f, viewport.y + viewport.height - corner}, 2.0f, border);
}

static void format_visible_faces(const int visible_faces[6], char *buffer, size_t size) {
    int wrote = 0;
    buffer[0] = '\0';
    for (int face_index = 0; face_index < 6; face_index++) {
        if (!visible_faces[face_index]) continue;
        if (wrote > 0) strncat(buffer, "/", size - strlen(buffer) - 1);
        strncat(buffer, FACE_NAMES[face_index], size - strlen(buffer) - 1);
        wrote++;
    }
    if (wrote == 0) strncpy(buffer, "None", size - 1);
    buffer[size - 1] = '\0';
}

static void append_text(char *buffer, size_t size, const char *text) {
    size_t used;
    size_t remaining;
    if (size == 0 || text == NULL) return;
    used = strlen(buffer);
    if (used >= size - 1) return;
    remaining = size - used - 1;
    strncat(buffer, text, remaining);
}

static void set_text_buffer(char *buffer, size_t size, const char *text) {
    if (size == 0 || buffer == NULL) return;
    if (text == NULL) {
        buffer[0] = '\0';
        return;
    }
    strncpy(buffer, text, size - 1);
    buffer[size - 1] = '\0';
}

static void apply_completion_response(char *input, size_t input_size, char *hint, size_t hint_size, const char *response) {
    const char *split;
    if (input == NULL || input_size == 0 || hint == NULL || hint_size == 0) return;
    if (response == NULL) {
        input[0] = '\0';
        hint[0] = '\0';
        return;
    }
    split = strchr(response, '\n');
    if (split == NULL) {
        set_text_buffer(input, input_size, response);
        hint[0] = '\0';
        return;
    }
    {
        size_t input_len = (size_t)(split - response);
        if (input_len >= input_size) input_len = input_size - 1;
        memcpy(input, response, input_len);
        input[input_len] = '\0';
        set_text_buffer(hint, hint_size, split + 1);
    }
}

static int wrapped_line_capacity(Rectangle rect, int font_size) {
    int line_height = font_size + 6;
    int visible_lines = (int)(rect.height / (float)line_height);
    return visible_lines < 1 ? 1 : visible_lines;
}

static size_t wrapped_line_span(const char *cursor, float max_width, int font_size) {
    char probe[512];
    size_t best = 0;
    size_t last_space = (size_t)-1;
    if (cursor == NULL || cursor[0] == '\0' || cursor[0] == '\n') return 0;
    for (size_t index = 0; cursor[index] != '\0' && cursor[index] != '\n' && index < sizeof(probe) - 1; index++) {
        probe[index] = cursor[index];
        probe[index + 1] = '\0';
        if (cursor[index] == ' ' || cursor[index] == '\t') last_space = index + 1;
        if (ui_measure_text(probe, font_size).x <= max_width) best = index + 1;
        else break;
    }
    if (best == 0) return 1;
    if (cursor[best] != '\0' && cursor[best] != '\n' && last_space != (size_t)-1) {
        size_t trimmed = last_space;
        while (trimmed > 0 && (cursor[trimmed - 1] == ' ' || cursor[trimmed - 1] == '\t')) trimmed--;
        if (trimmed > 0) best = trimmed;
    }
    return best;
}

static const char *advance_wrapped_cursor(const char *cursor, size_t span) {
    if (cursor == NULL || cursor[0] == '\0') return cursor;
    if (cursor[0] == '\n') return cursor + 1;
    cursor += span;
    if (cursor[0] == '\n') return cursor + 1;
    while (cursor[0] == ' ' || cursor[0] == '\t') cursor++;
    return cursor;
}

static int count_wrapped_lines(const char *text, float max_width, int font_size) {
    const char *cursor = text;
    int line_count = 0;
    if (text == NULL || text[0] == '\0') return 0;
    while (cursor[0] != '\0') {
        size_t span = wrapped_line_span(cursor, max_width, font_size);
        line_count++;
        cursor = advance_wrapped_cursor(cursor, span);
    }
    return line_count;
}

static int clamp_scroll_lines(int scroll_lines, int total_lines, int visible_lines) {
    int max_scroll = total_lines - visible_lines;
    if (max_scroll < 0) max_scroll = 0;
    if (scroll_lines < 0) scroll_lines = 0;
    if (scroll_lines > max_scroll) scroll_lines = max_scroll;
    return scroll_lines;
}

static void draw_wrapped_text_block(
    const char *text,
    Rectangle rect,
    int font_size,
    Color color,
    int scroll_lines,
    int *out_total_lines,
    int *out_visible_lines
) {
    const char *cursor = text;
    int line_height = font_size + 6;
    int line_index = 0;
    int drawn_lines = 0;
    int total_lines;
    int visible_lines = wrapped_line_capacity(rect, font_size);
    char line[512];

    if (out_visible_lines != NULL) *out_visible_lines = visible_lines;
    if (text == NULL || text[0] == '\0') {
        if (out_total_lines != NULL) *out_total_lines = 0;
        return;
    }

    total_lines = count_wrapped_lines(text, rect.width, font_size);
    scroll_lines = clamp_scroll_lines(scroll_lines, total_lines, visible_lines);
    if (out_total_lines != NULL) *out_total_lines = total_lines;

    BeginScissorMode((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height);
    while (cursor[0] != '\0' && drawn_lines < visible_lines) {
        size_t span = wrapped_line_span(cursor, rect.width, font_size);
        if (line_index >= scroll_lines) {
            size_t copy_length = span < sizeof(line) - 1 ? span : sizeof(line) - 1;
            memcpy(line, cursor, copy_length);
            line[copy_length] = '\0';
            ui_draw_text(line, rect.x, rect.y + (float)drawn_lines * (float)line_height, font_size, color);
            drawn_lines++;
        }
        line_index++;
        cursor = advance_wrapped_cursor(cursor, span);
    }
    EndScissorMode();
}

static Rectangle clamp_overlay_rect(Rectangle rect, int screen_w, int screen_h, float margin) {
    float max_width = (float)screen_w - margin * 2.0f;
    float max_height = (float)screen_h - margin * 2.0f;
    if (rect.width > max_width) rect.width = max_width;
    if (rect.height > max_height) rect.height = max_height;
    if (rect.x < margin) rect.x = margin;
    if (rect.y < margin) rect.y = margin;
    if (rect.x + rect.width > (float)screen_w - margin) rect.x = (float)screen_w - margin - rect.width;
    if (rect.y + rect.height > (float)screen_h - margin) rect.y = (float)screen_h - margin - rect.height;
    return rect;
}

static Rectangle default_help_panel(int screen_w, int screen_h) {
    return clamp_overlay_rect((Rectangle){(float)screen_w * 0.5f - 340.0f, (float)screen_h * 0.5f - 230.0f, 680.0f, 460.0f}, screen_w, screen_h, 18.0f);
}

static Rectangle help_body_rect(Rectangle panel) {
    return (Rectangle){panel.x + 20.0f, panel.y + 64.0f, panel.width - 40.0f, panel.height - 102.0f};
}

static Rectangle default_console_panel(int screen_w, int screen_h) {
    return clamp_overlay_rect((Rectangle){(float)screen_w * 0.5f - 360.0f, (float)screen_h * 0.5f - 190.0f, 720.0f, 380.0f}, screen_w, screen_h, 18.0f);
}

static Rectangle console_title_rect(Rectangle panel) {
    return (Rectangle){panel.x + 12.0f, panel.y + 12.0f, panel.width - 24.0f, 34.0f};
}

static Rectangle console_output_rect(Rectangle panel) {
    return (Rectangle){panel.x + 18.0f, panel.y + 98.0f, panel.width - 36.0f, panel.height - 160.0f};
}

static Rectangle console_input_rect(Rectangle panel) {
    return (Rectangle){panel.x + 18.0f, panel.y + panel.height - 50.0f, panel.width - 36.0f, 42.0f};
}

static void draw_hover_card(
    Rectangle viewport,
    int hover_has_cell,
    int hover_x,
    int hover_y,
    int hover_z,
    int hover_face,
    const void *grid_data,
    int dtype_bytes,
    int states,
    int H,
    int D
) {
    Rectangle card;
    char line1[128], line2[128], line3[128];
    uint64_t packed;
    int bits_per_face, face_value;
    if (!hover_has_cell || viewport.width < 260.0f || viewport.height < 120.0f) return;
    card = (Rectangle){viewport.x + 18.0f, viewport.y + 18.0f, 214.0f, 86.0f};
    packed = read_packed(grid_data, (size_t)hover_x * (size_t)(H * D) + (size_t)hover_y * (size_t)D + (size_t)hover_z, dtype_bytes);
    bits_per_face = ilog2_int(states);
    face_value = get_face_value(packed, hover_face, bits_per_face, states);
    draw_panel(card, ColorAlpha(PANEL_ALT, 0.92f), AMBER, NEON_MAGENTA);
    snprintf(line1, sizeof(line1), "HOVER");
    snprintf(line2, sizeof(line2), "Cell (%d, %d, %d)", hover_x, hover_y, hover_z);
    snprintf(line3, sizeof(line3), "%s face = %d", FACE_NAMES[hover_face], face_value);
    ui_draw_text(line1, card.x + 12.0f, card.y + 10.0f, 18, AMBER);
    ui_draw_text(line2, card.x + 12.0f, card.y + 36.0f, 18, TEXT);
    ui_draw_text(line3, card.x + 12.0f, card.y + 60.0f, 16, DIM);
}

static void draw_help_overlay(int screen_w, int screen_h, int scroll_lines) {
    Rectangle panel = default_help_panel(screen_w, screen_h);
    Rectangle body_rect = help_body_rect(panel);
    int total_lines = 0;
    int visible_lines = 0;
    int max_scroll = 0;

    draw_panel(panel, ColorAlpha(PANEL, 0.97f), CYAN, NEON_MAGENTA);
    ui_draw_text("HELP", panel.x + 20.0f, panel.y + 18.0f, 24, CYAN);
    ui_draw_text("Wheel or PgUp/PgDn scrolls this panel. Esc closes it.", panel.x + 20.0f, panel.y + 42.0f, 16, DIM);
    draw_wrapped_text_block(HELP_TEXT, body_rect, 17, TEXT, scroll_lines, &total_lines, &visible_lines);
    max_scroll = total_lines - visible_lines;
    if (max_scroll < 0) max_scroll = 0;
    if (max_scroll > 0) {
        char scroll_text[64];
        snprintf(scroll_text, sizeof(scroll_text), "Scroll %d / %d", scroll_lines, max_scroll);
        ui_draw_text(scroll_text, panel.x + 20.0f, panel.y + panel.height - 28.0f, 15, AMBER);
    } else {
        ui_draw_text("All help content is visible.", panel.x + 20.0f, panel.y + panel.height - 28.0f, 15, DIM);
    }
}

static void draw_console_overlay(
    Rectangle panel,
    const char *input,
    const char *output,
    int output_scroll,
    int last_ok,
    int has_selection,
    int sel_x,
    int sel_y,
    int sel_z,
    int sel_face,
    int has_hover,
    int hover_x,
    int hover_y,
    int hover_z,
    int hover_face
) {
    Rectangle title_rect = console_title_rect(panel);
    Rectangle output_rect = console_output_rect(panel);
    Rectangle input_rect = console_input_rect(panel);
    char context_text[256];
    char prompt_text[320];
    char scroll_text[64];
    const char *display_output = (output != NULL && output[0] != '\0') ? output : "No output yet. Try kube.inspect() or selected_cell";
    float cursor_on = sinf((float)GetTime() * 5.5f) > 0.0f ? 1.0f : 0.0f;
    int total_lines = 0;
    int visible_lines = 0;
    int max_scroll = 0;

    if (has_selection) snprintf(context_text, sizeof(context_text), "selection=(%d,%d,%d) face=%s", sel_x, sel_y, sel_z, FACE_NAMES[sel_face]);
    else strncpy(context_text, "selection=None", sizeof(context_text) - 1);
    context_text[sizeof(context_text) - 1] = '\0';
    if (has_hover) {
        append_text(context_text, sizeof(context_text), "  hover=(");
        {
            char hover_text[96];
            snprintf(hover_text, sizeof(hover_text), "%d,%d,%d) face=%s", hover_x, hover_y, hover_z, FACE_NAMES[hover_face]);
            append_text(context_text, sizeof(context_text), hover_text);
        }
    } else {
        append_text(context_text, sizeof(context_text), "  hover=None");
    }

    snprintf(prompt_text, sizeof(prompt_text), ">>> %s%s", input != NULL ? input : "", cursor_on > 0.0f ? "_" : "");
    draw_panel(panel, ColorAlpha(PANEL, 0.98f), CYAN, NEON_MAGENTA);
    draw_panel(title_rect, ColorAlpha(PANEL_ALT, 0.84f), CYAN, CYAN);
    draw_panel(output_rect, ColorAlpha(PANEL_ALT, 0.92f), last_ok ? CYAN : ERROR_C, last_ok ? CYAN : ERROR_C);
    draw_panel(input_rect, ColorAlpha(PANEL_ALT, 0.96f), AMBER, AMBER);
    ui_draw_text("PYTHON CONSOLE", panel.x + 18.0f, panel.y + 18.0f, 24, CYAN);
    ui_draw_text("Drag header to move. Enter: run  Wheel: scroll output  Esc/`/F2: close", panel.x + 18.0f, panel.y + 50.0f, 16, DIM);
    ui_draw_text(context_text, panel.x + 18.0f, panel.y + 74.0f, 16, AMBER);
    draw_wrapped_text_block(display_output, (Rectangle){output_rect.x + 12.0f, output_rect.y + 12.0f, output_rect.width - 24.0f, output_rect.height - 24.0f}, 16, TEXT, output_scroll, &total_lines, &visible_lines);
    max_scroll = total_lines - visible_lines;
    if (max_scroll < 0) max_scroll = 0;
    snprintf(scroll_text, sizeof(scroll_text), "Output %d / %d", output_scroll, max_scroll);
    ui_draw_text(scroll_text, output_rect.x + output_rect.width - ui_measure_text(scroll_text, 14).x - 12.0f, output_rect.y - 20.0f, 14, last_ok ? DIM : ERROR_C);
    ui_draw_text(prompt_text, input_rect.x + 12.0f, input_rect.y + 10.0f, 18, TEXT);
}

static void draw_cursor_fx(Vector2 mouse, Vector2 trail, int hot, float pulse) {
    Color primary = hot ? TEXT : CYAN;
    Color secondary = hot ? NEON_MAGENTA : AMBER;
    DrawCircleV(trail, 18.0f + 10.0f * pulse, ColorAlpha(primary, 0.08f));
    DrawRing(mouse, 8.0f + 5.0f * pulse, 13.0f + 10.0f * pulse, 0.0f, 360.0f, 36, ColorAlpha(primary, 0.72f));
    DrawCircleV(mouse, 2.6f, secondary);
    DrawLineEx((Vector2){mouse.x - 14.0f, mouse.y}, (Vector2){mouse.x - 5.0f, mouse.y}, 1.2f, ColorAlpha(primary, 0.54f));
    DrawLineEx((Vector2){mouse.x + 5.0f, mouse.y}, (Vector2){mouse.x + 14.0f, mouse.y}, 1.2f, ColorAlpha(primary, 0.54f));
    DrawLineEx((Vector2){mouse.x, mouse.y - 14.0f}, (Vector2){mouse.x, mouse.y - 5.0f}, 1.2f, ColorAlpha(primary, 0.54f));
    DrawLineEx((Vector2){mouse.x, mouse.y + 5.0f}, (Vector2){mouse.x, mouse.y + 14.0f}, 1.2f, ColorAlpha(primary, 0.54f));
}

static void draw_face_outline(Vector3 center, int face_index, Color color) {
    Vector3 verts[4];
    for (int i = 0; i < 4; i++) verts[i] = (Vector3){center.x + FACE_VERTS[face_index][i][0], center.y + FACE_VERTS[face_index][i][1], center.z + FACE_VERTS[face_index][i][2]};
    DrawLine3D(verts[0], verts[1], color);
    DrawLine3D(verts[1], verts[2], color);
    DrawLine3D(verts[2], verts[3], color);
    DrawLine3D(verts[3], verts[0], color);
}

static Vector3 cell_center(int x, int y, int z, int W, int H, int D) {
    return (Vector3){
        (float)x - (float)W / 2.0f + 0.5f,
        (float)y - (float)H / 2.0f + 0.5f,
        (float)z - (float)D / 2.0f + 0.5f
    };
}

static void draw_kube_outline(int W, int H, int D, const ViewConfig *view) {
    Color outline = view->view_mode == VIEW_SHELL ? ColorAlpha(CYAN, 0.74f) : ColorAlpha(TEXT, 0.74f);
    DrawCubeWiresV((Vector3){0.0f, 0.0f, 0.0f}, (Vector3){(float)W + 0.04f, (float)H + 0.04f, (float)D + 0.04f}, outline);
}

static void draw_visible_cell_grid(
    int W,
    int H,
    int D,
    const ViewConfig *view,
    const EditorState *editor,
    int hover_has_cell,
    int hover_x,
    int hover_y,
    int hover_z
) {
    Color grid_color = view->view_mode == VIEW_SHELL ? ColorAlpha(DIM, 0.26f) : ColorAlpha(TEXT, 0.18f);
    for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) for (int z = 0; z < D; z++) {
        int visible = count_faces_for_cell(x, y, z, W, H, D, view) > 0;
        int is_selected = editor->has_selection && x == editor->x && y == editor->y && z == editor->z;
        int is_hovered = hover_has_cell && x == hover_x && y == hover_y && z == hover_z;
        if (!visible || is_selected || is_hovered) continue;
        DrawCubeWiresV(cell_center(x, y, z, W, H, D), (Vector3){1.01f, 1.01f, 1.01f}, grid_color);
    }
}

static int try_edit_face(EditorState *editor, const ViewConfig *view, SetFaceFn set_face_cb, FlushFn flush_cb, int states, int new_value) {
    /* Apply one explicit face value chosen from UI or keyboard input. */
    if (!editor->has_selection) { set_status(editor, "Select a cell first"); return 0; }
    if (view->read_only) { set_status(editor, "Read-only mode"); return 0; }
    if (new_value < 0 || new_value >= states) { set_status(editor, "Value out of range"); return 0; }
    if (set_face_cb(editor->x, editor->y, editor->z, editor->face, new_value) == 0) { set_status(editor, "Edit failed"); return 0; }
    if (flush_cb != NULL && flush_cb() == 0) { set_status(editor, "Autosave failed"); return 0; }
    set_status(editor, "Autosaved");
    return 1;
}

static int try_edit_face_delta(const void *grid_data, int dtype_bytes, int states, EditorState *editor, const ViewConfig *view, SetFaceFn set_face_cb, FlushFn flush_cb, int delta, int H, int D) {
    /* Adjust the currently selected face relative to its current value. */
    int bits_per_face, current, next;
    size_t grid_index;
    if (!editor->has_selection) { set_status(editor, "Select a cell first"); return 0; }
    grid_index = (size_t)editor->x * (size_t)(H * D) + (size_t)editor->y * (size_t)D + (size_t)editor->z;
    bits_per_face = ilog2_int(states);
    current = get_face_value(read_packed(grid_data, grid_index, dtype_bytes), editor->face, bits_per_face, states);
    next = clamp_int(current + delta, 0, states - 1);
    return try_edit_face(editor, view, set_face_cb, flush_cb, states, next);
}

static int pick_cell(const Camera3D *camera, Rectangle viewport, const ViewConfig *view, int W, int H, int D, int *out_x, int *out_y, int *out_z, int *out_face) {
    /*
     * Ray-pick the best visible cell under the mouse cursor.
     *
     * This function exists because the editor needs a bridge from 2D cursor
     * position to 3D cell/face selection.
     */
    Vector2 mouse = GetMousePosition();
    Ray ray;
    float best_distance = 1000000.0f;
    int found = 0;
    if (!CheckCollisionPointRec(mouse, viewport)) return 0;
    ray = GetMouseRay(mouse, *camera);
    for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) for (int z = 0; z < D; z++) {
        int should_consider = 0;
        if (view->view_mode == VIEW_SLICE) should_consider = cell_in_slice(x, y, z, view->slice_axis, view->slice_index) && sample_slice_cell(x, y, z, W, H, D, view->slice_axis, view->lod);
        else for (int face_index = 0; face_index < 6; face_index++) {
            FaceBound bound = FACE_BOUND[face_index];
            int coords[3] = {x, y, z};
            int limits[3] = {W - 1, H - 1, D - 1};
            int limit = limits[bound.axis];
            int visible = bound.is_min ? (coords[bound.axis] == 0) : (coords[bound.axis] == limit);
            if (visible && view->visible_faces[face_index] && sample_shell_cell(x, y, z, W, H, D, face_index, view->lod)) { should_consider = 1; break; }
        }
        if (!should_consider) continue;
        {
            Vector3 center = {(float)x - (float)W / 2.0f + 0.5f,(float)y - (float)H / 2.0f + 0.5f,(float)z - (float)D / 2.0f + 0.5f};
            BoundingBox box = {{center.x - 0.5f, center.y - 0.5f, center.z - 0.5f},{center.x + 0.5f, center.y + 0.5f, center.z + 0.5f}};
            RayCollision collision = GetRayCollisionBox(ray, box);
            if (collision.hit && collision.distance < best_distance) {
                best_distance = collision.distance;
                *out_x = x; *out_y = y; *out_z = z; *out_face = face_from_normal(collision.normal);
                found = 1;
            }
        }
    }
    return found;
}

static void draw_selected_cell(int W, int H, int D, const EditorState *editor) {
    Vector3 center;
    if (!editor->has_selection) return;
    center = cell_center(editor->x, editor->y, editor->z, W, H, D);
    DrawCubeWiresV(center, (Vector3){1.06f,1.06f,1.06f}, TEXT);
    draw_face_outline(center, editor->face, NEON_MAGENTA);
}

static void draw_hover_cell(int W, int H, int D, int has_hover, int x, int y, int z, int face) {
    Vector3 center;
    if (!has_hover) return;
    center = cell_center(x, y, z, W, H, D);
    DrawCubeWiresV(center, (Vector3){1.03f,1.03f,1.03f}, ColorAlpha(AMBER, 0.92f));
    draw_face_outline(center, face, AMBER);
}

/*
 * SECTION_MAIN
 *
 * `kube_run(...)` is the exported entry point called from Python. It owns the
 * full window lifecycle, render loop, input handling, geometry rebuild logic,
 * and shutdown path.
 */
KUBE_API void kube_run(
    const void *grid_data,
    int W,
    int H,
    int D,
    int states,
    int dtype_bytes,
    int view_mode,
    int slice_axis,
    int slice_index,
    int lod,
    int window_width,
    int window_height,
    int read_only,
    const char *file_label,
    const char *storage_label,
    SetFaceFn set_face_cb,
    FlushFn flush_cb,
    RunCodeFn run_code_cb
) {
    ViewConfig view;
    EditorState editor;
    Camera3D camera;
    KubeUiState *ui = NULL;
    KubeUiFrameDesc ui_frame;
    KubeUiActions ui_actions;
    GPU gpu = {0};
    Geo geo = {0};
    int current_visible_faces[6] = {0,0,0,0,0,0};
    int show_grid = 1;
    int show_help = 0;
    int show_console = 0;
    int console_last_ok = 1;
    const float default_yaw = 32.0f;
    const float default_pitch = 22.0f;
    const float default_distance = (float)(W > H ? (W > D ? W : D) : (H > D ? H : D)) * 2.3f;
    float yaw = default_yaw;
    float pitch = default_pitch;
    float distance = default_distance;
    int hover_has_cell = 0;
    int hover_x = 0;
    int hover_y = 0;
    int hover_z = 0;
    int hover_face = 0;
    int needs_rebuild = 1;
    int cursor_initialized = 0;
    Vector2 cursor_trail = {0};
    float cursor_pulse = 0.0f;
    float ui_intro = 0.0f;
    char console_input[256] = {0};
    char console_hint[2048] = "tab completes names like kube., selection, and inspect_selection";
    char console_output[8192] = "Ready. Try kube.inspect(), selected_cell, or face(0, 0, 0, 0)";
    char console_last_command[256] = {0};
    char console_complete_response[4096] = {0};

    memset(&editor, 0, sizeof(editor));
    strcpy(editor.status, "Ready");
    editor.face = 0;

    view.view_mode = view_mode;
    view.slice_axis = clamp_int(slice_axis, 0, 2);
    view.slice_index = clamp_int(slice_index, 0, view.slice_axis == 0 ? W - 1 : (view.slice_axis == 1 ? H - 1 : D - 1));
    view.lod = lod < 1 ? 1 : lod;
    view.read_only = read_only ? 1 : 0;
    for (int i = 0; i < 6; i++) view.visible_faces[i] = 0;

    if (window_width < 960) window_width = 960;
    if (window_height < 640) window_height = 640;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(window_width, window_height, "Kube Neon Editor");
    SetWindowMinSize(960, 640);
    SetTargetFPS(60);
    init_ui_font();
    ui = kube_ui_create(UI_FONT, window_width, window_height);
    if (ui == NULL) {
        shutdown_ui_font();
        CloseWindow();
        return;
    }
    HideCursor();
    camera = (Camera3D){0};
    camera.up = (Vector3){0.0f,1.0f,0.0f};
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    while (!WindowShouldClose()) {
        float delta_time = GetFrameTime();
        int screen_w = GetScreenWidth();
        int screen_h = GetScreenHeight();
        Vector2 mouse = GetMousePosition();
        float wheel = GetMouseWheelMove();
        Rectangle viewport;
        uint64_t selected_packed = 0;
        int selected_face_values[6] = {0,0,0,0,0,0};
        int hover_face_value = 0;
        int over_viewport = 0;
        int edited_this_frame = 0;
        int rendered_faces = gpu.n_idx / 6;

        ui_intro = approachf(ui_intro, 1.0f, delta_time * 2.6f);
        if (editor.has_selection) {
            int bits_per_face = ilog2_int(states);
            size_t grid_index = (size_t)editor.x * (size_t)(H * D) + (size_t)editor.y * (size_t)D + (size_t)editor.z;
            selected_packed = read_packed(grid_data, grid_index, dtype_bytes);
            for (int face_index = 0; face_index < 6; face_index++) {
                selected_face_values[face_index] = get_face_value(selected_packed, face_index, bits_per_face, states);
            }
        }
        if (hover_has_cell) {
            int bits_per_face = ilog2_int(states);
            size_t hover_index = (size_t)hover_x * (size_t)(H * D) + (size_t)hover_y * (size_t)D + (size_t)hover_z;
            hover_face_value = get_face_value(read_packed(grid_data, hover_index, dtype_bytes), hover_face, bits_per_face, states);
        }
        memset(&ui_frame, 0, sizeof(ui_frame));
        ui_frame.screen_w = screen_w;
        ui_frame.screen_h = screen_h;
        ui_frame.delta_time = delta_time;
        ui_frame.mouse = mouse;
        ui_frame.wheel = wheel;
        ui_frame.mouse_left_down = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
        ui_frame.mouse_left_pressed = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        ui_frame.page_up_pressed = IsKeyPressed(KEY_PAGE_UP);
        ui_frame.page_down_pressed = IsKeyPressed(KEY_PAGE_DOWN);
        ui_frame.show_help = show_help;
        ui_frame.show_console = show_console;
        ui_frame.show_grid = show_grid;
        ui_frame.console_last_ok = console_last_ok;
        ui_frame.view_mode = view.view_mode;
        ui_frame.slice_axis = view.slice_axis;
        ui_frame.slice_index = view.slice_index;
        ui_frame.lod = view.lod;
        ui_frame.read_only = view.read_only;
        ui_frame.W = W;
        ui_frame.H = H;
        ui_frame.D = D;
        ui_frame.states = states;
        ui_frame.rendered_faces = rendered_faces;
        ui_frame.visible_faces = view.visible_faces;
        ui_frame.file_label = file_label;
        ui_frame.storage_label = storage_label;
        ui_frame.status_text = editor.status;
        ui_frame.status_visible = editor.status_frames > 0;
        ui_frame.editor_has_selection = editor.has_selection;
        ui_frame.sel_x = editor.x;
        ui_frame.sel_y = editor.y;
        ui_frame.sel_z = editor.z;
        ui_frame.sel_face = editor.face;
        ui_frame.selected_packed = selected_packed;
        ui_frame.selected_face_values = selected_face_values;
        ui_frame.hover_has_cell = hover_has_cell;
        ui_frame.hover_x = hover_x;
        ui_frame.hover_y = hover_y;
        ui_frame.hover_z = hover_z;
        ui_frame.hover_face = hover_face;
        ui_frame.hover_face_value = hover_face_value;
        ui_frame.console_input = console_input;
        ui_frame.console_hint = console_hint;
        ui_frame.console_output = console_output;
        kube_ui_update(ui, &ui_frame, &ui_actions);
        viewport = ui_actions.viewport_rect;
        over_viewport = CheckCollisionPointRec(mouse, viewport);
        if (!cursor_initialized) { cursor_trail = mouse; cursor_initialized = 1; }
        cursor_trail.x = approachf(cursor_trail.x, mouse.x, delta_time * 12.0f);
        cursor_trail.y = approachf(cursor_trail.y, mouse.y, delta_time * 12.0f);
        cursor_pulse = approachf(cursor_pulse, 0.0f, delta_time * 5.5f);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) cursor_pulse = 1.0f;

        if (show_console && ui_actions.clicked_console_close) {
            show_console = 0;
            set_status(&editor, "Python console closed");
        }

        if (IsKeyPressed(KEY_GRAVE) || IsKeyPressed(KEY_F2)) {
            show_console = !show_console;
            if (show_console) {
                show_help = 0;
                set_status(&editor, "Python console opened");
            } else {
                set_status(&editor, "Python console closed");
            }
        }
        if ((IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL)) && IsKeyPressed(KEY_Q)) {
            break;
        }
        if (IsKeyPressed(KEY_ESCAPE)) {
            if (show_console) {
                show_console = 0;
                set_status(&editor, "Python console closed");
            } else if (show_help) {
                show_help = 0;
                set_status(&editor, "Help overlay closed");
            }
        }
        if (!show_console && (IsKeyPressed(KEY_F1) || IsKeyPressed(KEY_H))) {
            show_help = !show_help;
            set_status(&editor, show_help ? "Help overlay opened" : "Help overlay closed");
        }
        if (!show_console && IsKeyPressed(KEY_G)) {
            show_grid = !show_grid;
            set_status(&editor, show_grid ? "Grid overlay enabled" : "Grid overlay hidden");
        }
        if (!show_console && IsKeyPressed(KEY_R)) {
            yaw = default_yaw;
            pitch = default_pitch;
            distance = default_distance;
            set_status(&editor, "Camera reset");
        }
        if (!show_console && (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_DELETE)) && editor.has_selection) {
            editor.has_selection = 0;
            set_status(&editor, "Selection cleared");
        }
        if (show_console) {
            int codepoint = GetCharPressed();
            size_t input_length = strlen(console_input);
            while (codepoint > 0) {
                if (codepoint >= 32 && codepoint <= 126 && input_length < sizeof(console_input) - 1) {
                    console_input[input_length++] = (char)codepoint;
                    console_input[input_length] = '\0';
                    console_hint[0] = '\0';
                }
                codepoint = GetCharPressed();
            }
            if (IsKeyPressed(KEY_BACKSPACE) && input_length > 0) {
                console_input[input_length - 1] = '\0';
                input_length--;
                console_hint[0] = '\0';
            }
            if ((IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL)) && IsKeyPressed(KEY_V)) {
                const char *clipboard = GetClipboardText();
                if (clipboard != NULL) {
                    append_text(console_input, sizeof(console_input), clipboard);
                    console_hint[0] = '\0';
                }
            }
            if (IsKeyPressed(KEY_UP) && console_last_command[0] != '\0' && console_input[0] == '\0') {
                strncpy(console_input, console_last_command, sizeof(console_input) - 1);
                console_input[sizeof(console_input) - 1] = '\0';
                set_text_buffer(console_hint, sizeof(console_hint), "history recalled");
            }
            if (IsKeyPressed(KEY_TAB) && run_code_cb != NULL) {
                char complete_request[512];
                snprintf(complete_request, sizeof(complete_request), "%s%s", CONSOLE_COMPLETE_PREFIX, console_input);
                console_complete_response[0] = '\0';
                if (run_code_cb(
                    complete_request,
                    editor.has_selection,
                    editor.x,
                    editor.y,
                    editor.z,
                    editor.face,
                    hover_has_cell,
                    hover_x,
                    hover_y,
                    hover_z,
                    hover_face,
                    console_complete_response,
                    (int)sizeof(console_complete_response)
                )) {
                    apply_completion_response(
                        console_input,
                        sizeof(console_input),
                        console_hint,
                        sizeof(console_hint),
                        console_complete_response
                    );
                } else {
                    set_text_buffer(console_hint, sizeof(console_hint), "completion unavailable");
                }
            }
            if (IsKeyPressed(KEY_ENTER) && console_input[0] != '\0') {
                strncpy(console_last_command, console_input, sizeof(console_last_command) - 1);
                console_last_command[sizeof(console_last_command) - 1] = '\0';
                if (run_code_cb != NULL) {
                    console_last_ok = run_code_cb(
                        console_input,
                        editor.has_selection,
                        editor.x,
                        editor.y,
                        editor.z,
                        editor.face,
                        hover_has_cell,
                        hover_x,
                        hover_y,
                        hover_z,
                        hover_face,
                        console_output,
                        (int)sizeof(console_output)
                    );
                    needs_rebuild = 1;
                    console_hint[0] = '\0';
                    set_status(&editor, console_last_ok ? "Python executed" : "Python error");
                } else {
                    strncpy(console_output, "Python console callback unavailable", sizeof(console_output) - 1);
                    console_output[sizeof(console_output) - 1] = '\0';
                    console_last_ok = 0;
                    set_status(&editor, "Python console unavailable");
                }
                console_input[0] = '\0';
            }
        }
        if (!show_help && !show_console && IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && over_viewport) {
            Vector2 delta = GetMouseDelta();
            yaw += delta.x * 0.35f;
            pitch += delta.y * 0.35f;
            pitch = (float)clamp_int((int)pitch, -89, 89);
        }
        if (!show_help && !show_console && !ui_actions.pointer_over_ui) distance -= wheel * 0.7f;
        if (distance < 1.0f) distance = 1.0f;

        if (!show_help && !show_console && IsKeyPressed(KEY_TAB)) { view.view_mode = view.view_mode == VIEW_SHELL ? VIEW_SLICE : VIEW_SHELL; needs_rebuild = 1; set_status(&editor, view.view_mode == VIEW_SHELL ? "Shell mode" : "Slice mode"); }
        if (!show_help && !show_console && IsKeyPressed(KEY_X)) { view.slice_axis = 0; view.slice_index = clamp_int(view.slice_index, 0, W - 1); needs_rebuild = 1; }
        if (!show_help && !show_console && IsKeyPressed(KEY_Y)) { view.slice_axis = 1; view.slice_index = clamp_int(view.slice_index, 0, H - 1); needs_rebuild = 1; }
        if (!show_help && !show_console && IsKeyPressed(KEY_Z)) { view.slice_axis = 2; view.slice_index = clamp_int(view.slice_index, 0, D - 1); needs_rebuild = 1; }
        if (!show_help && !show_console && IsKeyPressed(KEY_PAGE_UP)) { int limit = view.slice_axis == 0 ? W - 1 : (view.slice_axis == 1 ? H - 1 : D - 1); view.slice_index = clamp_int(view.slice_index + 1, 0, limit); needs_rebuild = 1; }
        if (!show_help && !show_console && IsKeyPressed(KEY_PAGE_DOWN)) { int limit = view.slice_axis == 0 ? W - 1 : (view.slice_axis == 1 ? H - 1 : D - 1); view.slice_index = clamp_int(view.slice_index - 1, 0, limit); needs_rebuild = 1; }
        if (!show_help && !show_console && (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT))) { view.lod = clamp_int(view.lod + 1, 1, 32); needs_rebuild = 1; }
        if (!show_help && !show_console && (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))) { view.lod = clamp_int(view.lod - 1, 1, 32); needs_rebuild = 1; }
        if (!show_help && !show_console) for (int face_index = 0; face_index < 6; face_index++) if (IsKeyPressed(KEY_ONE + face_index)) editor.face = face_index;

        {
            float yaw_radians = yaw * DEG2RAD;
            float pitch_radians = pitch * DEG2RAD;
            camera.position = (Vector3){distance * cosf(pitch_radians) * sinf(yaw_radians), distance * sinf(pitch_radians), distance * cosf(pitch_radians) * cosf(yaw_radians)};
            camera.target = (Vector3){0.0f,0.0f,0.0f};
        }

        if (view.view_mode == VIEW_SHELL) {
            int desired_faces[6];
            update_visible_faces(&camera, desired_faces);
            if (visible_faces_changed(current_visible_faces, desired_faces)) {
                for (int i = 0; i < 6; i++) { current_visible_faces[i] = desired_faces[i]; view.visible_faces[i] = desired_faces[i]; }
                needs_rebuild = 1;
            }
        }
        else {
            for (int i = 0; i < 6; i++) view.visible_faces[i] = 1;
        }

        if (!show_console) hover_has_cell = 0;
        if (!show_help && !show_console && over_viewport && !ui_actions.pointer_over_ui && !IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            hover_has_cell = pick_cell(&camera, viewport, &view, W, H, D, &hover_x, &hover_y, &hover_z, &hover_face);
            if (hover_has_cell && view.view_mode == VIEW_SLICE) hover_face = slice_face_for_camera(&camera, view.slice_axis);
        }

        if (!show_help && !show_console) {
            if (ui_actions.clicked_shell) { view.view_mode = VIEW_SHELL; needs_rebuild = 1; }
            else if (ui_actions.clicked_slice) { view.view_mode = VIEW_SLICE; needs_rebuild = 1; }
            else if (ui_actions.clicked_axis >= 0) {
                view.slice_axis = ui_actions.clicked_axis;
                view.slice_index = clamp_int(view.slice_index, 0, ui_actions.clicked_axis == 0 ? W - 1 : (ui_actions.clicked_axis == 1 ? H - 1 : D - 1));
                needs_rebuild = 1;
            } else if (ui_actions.clicked_slice_delta != 0) {
                int limit = view.slice_axis == 0 ? W - 1 : (view.slice_axis == 1 ? H - 1 : D - 1);
                view.slice_index = clamp_int(view.slice_index + ui_actions.clicked_slice_delta, 0, limit);
                needs_rebuild = 1;
            } else if (ui_actions.clicked_lod_delta != 0) {
                view.lod = clamp_int(view.lod + ui_actions.clicked_lod_delta, 1, 32);
                needs_rebuild = 1;
            }

            if (ui_actions.clicked_face_row >= 0) editor.face = ui_actions.clicked_face_row;
            if (ui_actions.clicked_face_adjust_row >= 0) {
                editor.face = ui_actions.clicked_face_adjust_row;
                edited_this_frame = try_edit_face_delta(grid_data, dtype_bytes, states, &editor, &view, set_face_cb, flush_cb, ui_actions.clicked_face_adjust_delta, H, D);
            }
            if (ui_actions.clicked_chip_value >= 0) edited_this_frame = try_edit_face(&editor, &view, set_face_cb, flush_cb, states, ui_actions.clicked_chip_value);

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !ui_actions.pointer_over_ui && hover_has_cell) {
                editor.has_selection = 1;
                editor.x = hover_x; editor.y = hover_y; editor.z = hover_z; editor.face = hover_face;
                set_status(&editor, "Cell selected");
            }
        }

        if (!show_help && !show_console && editor.has_selection) {
            if (IsKeyPressed(KEY_LEFT_BRACKET)) edited_this_frame = try_edit_face_delta(grid_data, dtype_bytes, states, &editor, &view, set_face_cb, flush_cb, -1, H, D);
            if (IsKeyPressed(KEY_RIGHT_BRACKET)) edited_this_frame = try_edit_face_delta(grid_data, dtype_bytes, states, &editor, &view, set_face_cb, flush_cb, +1, H, D);
            if (states <= 10) {
                int digit_keys[10] = {KEY_KP_0,KEY_KP_1,KEY_KP_2,KEY_KP_3,KEY_KP_4,KEY_KP_5,KEY_KP_6,KEY_KP_7,KEY_KP_8,KEY_KP_9};
                for (int value = 0; value < states; value++) if (IsKeyPressed(digit_keys[value])) edited_this_frame = try_edit_face(&editor, &view, set_face_cb, flush_cb, states, value);
                if (IsKeyPressed(KEY_ZERO)) edited_this_frame = try_edit_face(&editor, &view, set_face_cb, flush_cb, states, 0);
                for (int value = 7; value < states; value++) if (IsKeyPressed(KEY_ZERO + value)) edited_this_frame = try_edit_face(&editor, &view, set_face_cb, flush_cb, states, value);
            }
        }
        if (edited_this_frame) needs_rebuild = 1;

        if (needs_rebuild) {
            gpu_free(&gpu); geo_free(&geo);
            geo = build_geometry(grid_data, W, H, D, states, dtype_bytes, &view);
            if (geo.n_idx > 0) gpu = upload_geo(&geo);
            geo_free(&geo);
            needs_rebuild = 0;
        }

        BeginDrawing();
        ui_begin_frame();
        ClearBackground(BG);
        draw_scanlines(screen_w, screen_h);
        BeginMode3D(camera);
        if (gpu.vao != 0) { rlEnableVertexArray(gpu.vao); rlDrawVertexArrayElements(0, gpu.n_idx, NULL); rlDisableVertexArray(); }
        draw_kube_outline(W, H, D, &view);
        if (show_grid) draw_visible_cell_grid(W, H, D, &view, &editor, hover_has_cell, hover_x, hover_y, hover_z);
        draw_hover_cell(W, H, D, hover_has_cell && !editor.has_selection, hover_x, hover_y, hover_z, hover_face);
        if (hover_has_cell && editor.has_selection && (hover_x != editor.x || hover_y != editor.y || hover_z != editor.z || hover_face != editor.face)) draw_hover_cell(W, H, D, 1, hover_x, hover_y, hover_z, hover_face);
        draw_selected_cell(W, H, D, &editor);
        EndMode3D();
        draw_viewport_overlay(viewport, hover_has_cell, ui_intro);
        kube_ui_render(ui);
        if (editor.status_frames > 0) editor.status_frames--;
        draw_cursor_fx(mouse, cursor_trail, ui_actions.pointer_over_ui || (hover_has_cell && over_viewport), cursor_pulse);
        EndDrawing();
    }

    gpu_free(&gpu);
    kube_ui_destroy(ui);
    shutdown_ui_font();
    ShowCursor();
    CloseWindow();
}
