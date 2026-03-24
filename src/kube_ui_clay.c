/*
 * kube_ui_clay.c - Clay-backed editor chrome for the native Kube renderer
 *
 * This file exists to keep layout, wrapping, floating panels, and hit-tested UI
 * behavior out of the main renderer loop.
 *
 * Responsibilities here:
 * - define the visual system for the editor overlay
 * - declare Clay layouts for help, inspector, console, and bars
 * - translate editor/runtime state into UI text and component state
 * - return high-level UI actions back to `kube_renderer.c`
 *
 * Non-responsibilities:
 * - 3D kube geometry and camera rendering
 * - persistence or packed-buffer mutation
 * - Python callback execution
 */

#include "kube_ui_clay.h"

#define CLAY_IMPLEMENTATION
#include "clay.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VIEW_SHELL = 0, VIEW_SLICE = 1 };

/* Fixed chrome dimensions that define the terminal-style overlay proportions. */
enum {
    KUBE_UI_TOP_BAR_H = 58,
    KUBE_UI_BOTTOM_BAR_H = 92,
    KUBE_UI_INSPECTOR_W = 320,
    KUBE_UI_HELP_W = 680,
    KUBE_UI_HELP_H = 460,
    KUBE_UI_CONSOLE_MIN_W = 380,
    KUBE_UI_CONSOLE_MAX_W = 460,
    KUBE_UI_CONSOLE_MIN_H = 280,
};

typedef struct HelpSection {
    const char *title;
    const char *body;
} HelpSection;

static const HelpSection HELP_SECTIONS[] = {
    {"Navigation", "Right drag orbits the kube. Mouse wheel zooms. R resets the camera."},
    {"Views", "Tab toggles shell and slice mode. X, Y, and Z switch slice axis. PgUp and PgDn move the active slice. Plus and minus adjust LOD."},
    {"Selection", "Left click picks a visible cell. Keys 1 through 6 switch the active face. Backspace or Delete clears the current selection."},
    {"Editing", "Left bracket and right bracket adjust the selected face value. Numpad digits assign explicit values when the state count is 10 or less."},
    {"Debug", "Backtick or F2 opens the Python console. G toggles the grid. H or F1 opens help. Escape closes overlays. Ctrl+Q exits the editor."},
    {"Console", "The console is draggable from its header. Tab autocompletes Python names. The output pane scrolls with the mouse wheel and PgUp or PgDn."},
};

typedef struct KubeUiState {
    Font font;
    Clay_Context *context;
    void *arena_memory;
    size_t arena_capacity;
    Clay_RenderCommandArray commands;
    Rectangle viewport_rect;
    Rectangle inspector_rect;
    Vector2 console_position;
    Vector2 console_drag_offset;
    int console_position_initialized;
    int console_dragging;
    int last_show_console;
    char top_line1[256];
    char top_line2[256];
    char slice_label[32];
    char lod_label[32];
    char hover_line1[128];
    char hover_line2[128];
    char inspector_title[128];
    char inspector_packed[128];
    char inspector_state[64];
    char console_context[256];
    char console_prompt[320];
    char status_text[160];
    char face_labels[6][64];
    char chip_labels[10][4];
    char last_error[256];
} KubeUiState;

/* Monochrome terminal palette with a few restrained accents for console states. */
static const Color BG = {5,5,5,255};
static const Color PANEL = {10,10,10,244};
static const Color PANEL_ALT = {18,18,18,236};
static const Color PANEL_SOFT = {26,26,26,228};
static const Color LINE = {238,238,238,255};
static const Color LINE_SOFT = {120,120,120,255};
static const Color LINE_FAINT = {74,74,74,255};
static const Color TEXT = {244,244,244,255};
static const Color DIM = {156,156,156,255};
static const Color SUBTLE = {102,102,102,255};
static const Color INVERSE = {250,250,250,255};
static const Color INK = {6,6,6,255};
static const Color SUCCESS = {232,232,232,255};
static const Color ERROR_C = {182,182,182,255};
static const Color TERM_GREEN = {168,198,146,255};
static const Color TERM_AMBER = {208,182,116,255};
static const Color TERM_RED = {182,126,126,255};
static const Color TERM_BLUE = {144,162,184,255};

/* Small color/string helpers used throughout the Clay tree declarations. */
static Clay_Color clay_color(Color color) {
    return (Clay_Color){(float)color.r, (float)color.g, (float)color.b, (float)color.a};
}

static Clay_String clay_string(const char *text) {
    Clay_String string = {0};
    if (text == NULL) return string;
    string.chars = text;
    string.length = (int32_t)strlen(text);
    string.isStaticallyAllocated = false;
    return string;
}

static float clampf_local(float value, float min_value, float max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

/* The console intentionally stays narrow and tall like a floating terminal pane. */
static float console_panel_width(int screen_w) {
    float width = (float)screen_w * 0.34f;
    if (width < KUBE_UI_CONSOLE_MIN_W) width = KUBE_UI_CONSOLE_MIN_W;
    if (width > KUBE_UI_CONSOLE_MAX_W) width = KUBE_UI_CONSOLE_MAX_W;
    return width;
}

static float console_panel_height(int screen_h) {
    float height = floorf((float)screen_h * 0.5f);
    float max_height = (float)screen_h - 36.0f;
    if (height < KUBE_UI_CONSOLE_MIN_H) height = KUBE_UI_CONSOLE_MIN_H;
    if (height > max_height) height = max_height;
    return height;
}

static int point_in_box(Vector2 point, Clay_BoundingBox box) {
    return point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height;
}

static Clay_TextElementConfig *ui_text_config(Color color, uint16_t font_size, Clay_TextElementConfigWrapMode wrap_mode) {
    return CLAY_TEXT_CONFIG({
        .textColor = clay_color(color),
        .fontSize = font_size,
        .lineHeight = (uint16_t)(font_size + 6),
        .wrapMode = wrap_mode
    });
}

static Clay_Dimensions kube_ui_measure_text(Clay_StringSlice text, Clay_TextElementConfig *config, void *user_data) {
    KubeUiState *state = (KubeUiState *)user_data;
    Font font = state->font;
    int font_size = config != NULL && config->fontSize > 0 ? config->fontSize : 18;
    float spacing = config != NULL ? (float)config->letterSpacing : 1.0f;
    Clay_Dimensions dimensions = {0};
    char stack_buffer[1024];
    char *buffer = stack_buffer;

    if (text.length < 0) return dimensions;
    if ((size_t)text.length + 1 > sizeof(stack_buffer)) {
        buffer = (char *)malloc((size_t)text.length + 1);
        if (buffer == NULL) return dimensions;
    }
    memcpy(buffer, text.chars, (size_t)text.length);
    buffer[text.length] = '\0';
    {
        Vector2 measured = MeasureTextEx(font, buffer, (float)font_size, spacing);
        dimensions.width = measured.x;
        dimensions.height = config != NULL && config->lineHeight > 0 ? (float)config->lineHeight : measured.y;
    }
    if (buffer != stack_buffer) free(buffer);
    return dimensions;
}

static void kube_ui_error_handler(Clay_ErrorData error_data) {
    KubeUiState *state = (KubeUiState *)error_data.userData;
    int length = error_data.errorText.length;
    if (state == NULL) return;
    if (length < 0) length = 0;
    if (length >= (int)sizeof(state->last_error)) length = (int)sizeof(state->last_error) - 1;
    memcpy(state->last_error, error_data.errorText.chars, (size_t)length);
    state->last_error[length] = '\0';
}

static Color color_for_state(int state, int states) {
    unsigned char value;
    if (states <= 1) return LINE_SOFT;
    value = (unsigned char)(34 + (state * 196) / (states - 1));
    return (Color){value, value, value, 255};
}

static Color text_for_fill(Color fill) {
    int luminance = (fill.r * 299 + fill.g * 587 + fill.b * 114) / 1000;
    return luminance >= 150 ? INK : TEXT;
}

static void format_visible_faces(const int visible_faces[6], char *buffer, size_t size) {
    static const char *face_names[6] = {"Top","Bottom","Left","Right","Front","Back"};
    int wrote = 0;
    buffer[0] = '\0';
    for (int face_index = 0; face_index < 6; face_index++) {
        if (!visible_faces[face_index]) continue;
        if (wrote > 0) strncat(buffer, "/", size - strlen(buffer) - 1);
        strncat(buffer, face_names[face_index], size - strlen(buffer) - 1);
        wrote++;
    }
    if (wrote == 0) strncpy(buffer, "None", size - 1);
    buffer[size - 1] = '\0';
}

static Clay_BorderElementConfig border_config(Color color, uint16_t width) {
    Clay_BorderElementConfig border = {0};
    border.color = clay_color(color);
    border.width.left = width;
    border.width.right = width;
    border.width.top = width;
    border.width.bottom = width;
    border.width.betweenChildren = width;
    return border;
}

static Clay_CornerRadius rounded_panel_radius(void) {
    return CLAY_CORNER_RADIUS(0);
}

static void ui_button(Clay_ElementId id, const char *label, float width, Color fill, Color outline, int active) {
    int hovered = Clay_Hovered();
    Color bg;
    Color border;
    Color text_color;

    (void)fill;
    (void)outline;
    bg = active ? INVERSE : (hovered ? PANEL_SOFT : PANEL_ALT);
    border = active ? INVERSE : (hovered ? LINE : LINE_SOFT);
    text_color = active ? INK : TEXT;
    CLAY(id, {
        .layout = {
            .sizing = { .width = CLAY_SIZING_FIXED(width), .height = CLAY_SIZING_FIXED(30) },
            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
        },
        .backgroundColor = clay_color(bg),
        .cornerRadius = rounded_panel_radius(),
        .border = border_config(border, active ? 2 : 1)
    }) {
        CLAY_TEXT(clay_string(label), ui_text_config(text_color, 17, CLAY_TEXT_WRAP_NONE));
    }
}

static void ui_chip(Clay_ElementId id, const char *label, Color fill, int active, float width) {
    int hovered = Clay_Hovered();
    Color border = active ? INVERSE : (hovered ? LINE : LINE_SOFT);
    Color bg = hovered && !active ? ColorAlpha(fill, 0.92f) : fill;
    CLAY(id, {
        .layout = {
            .sizing = { .width = CLAY_SIZING_FIXED(width), .height = CLAY_SIZING_FIXED(24) },
            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
        },
        .backgroundColor = clay_color(bg),
        .cornerRadius = CLAY_CORNER_RADIUS(0),
        .border = border_config(border, active ? 2 : 1)
    }) {
        CLAY_TEXT(clay_string(label), ui_text_config(text_for_fill(bg), 14, CLAY_TEXT_WRAP_NONE));
    }
}

static void build_top_bar(KubeUiState *state, const KubeUiFrameDesc *frame) {
    CLAY(CLAY_ID("TopBar"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(KUBE_UI_TOP_BAR_H) },
            .padding = {16, 16, 10, 8},
            .childGap = 12,
            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
        },
        .backgroundColor = clay_color(PANEL),
        .border = border_config(LINE_SOFT, 1)
    }) {
        CLAY(CLAY_ID("TopBar.Left"), {
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) }, .layoutDirection = CLAY_TOP_TO_BOTTOM }
        }) {
            CLAY_TEXT(clay_string(state->top_line1), ui_text_config(TEXT, 18, CLAY_TEXT_WRAP_WORDS));
            CLAY_TEXT(clay_string(state->top_line2), ui_text_config(frame->read_only ? TEXT : DIM, 15, CLAY_TEXT_WRAP_WORDS));
        }
        CLAY(CLAY_ID("TopBar.Status"), {
            .layout = { .sizing = { .width = CLAY_SIZING_FIT(0), .height = CLAY_SIZING_FIT(0) }, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } }
        }) {
            if (frame->status_visible) {
                Color status_bg = strstr(state->status_text, "failed") != NULL ? LINE_SOFT : INVERSE;
                Color status_text = status_bg.r > 180 ? INK : TEXT;
                CLAY(CLAY_ID("TopBar.StatusBadge"), {
                    .layout = {
                        .padding = {10, 10, 7, 7}
                    },
                    .backgroundColor = clay_color(status_bg),
                    .border = border_config(status_bg, 1)
                }) {
                    CLAY_TEXT(clay_string(state->status_text), ui_text_config(status_text, 15, CLAY_TEXT_WRAP_NONE));
                }
            }
        }
    }
}

static void build_bottom_bar(KubeUiState *state, const KubeUiFrameDesc *frame) {
    CLAY(CLAY_ID("BottomBar"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(KUBE_UI_BOTTOM_BAR_H) },
            .padding = {16, 16, 12, 12},
            .childGap = 16,
            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
        },
        .backgroundColor = clay_color(PANEL),
        .border = border_config(LINE_SOFT, 1)
    }) {
        CLAY(CLAY_ID("BottomBar.Controls"), {
            .layout = { .childGap = 8, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } }
        }) {
            ui_button(CLAY_ID("Mode.Shell"), "SHELL", 92, PANEL_ALT, LINE, frame->view_mode == VIEW_SHELL);
            ui_button(CLAY_ID("Mode.Slice"), "SLICE", 92, PANEL_ALT, LINE, frame->view_mode == VIEW_SLICE);
            ui_button(CLAY_ID("Axis.X"), "X", 42, PANEL_ALT, LINE, frame->slice_axis == 0);
            ui_button(CLAY_ID("Axis.Y"), "Y", 42, PANEL_ALT, LINE, frame->slice_axis == 1);
            ui_button(CLAY_ID("Axis.Z"), "Z", 42, PANEL_ALT, LINE, frame->slice_axis == 2);
            ui_button(CLAY_ID("Slice.Prev"), "<", 34, PANEL, LINE_SOFT, 0);
            ui_button(CLAY_ID("Slice.Next"), ">", 34, PANEL, LINE_SOFT, 0);
            ui_button(CLAY_ID("Lod.Down"), "-", 34, PANEL, LINE_SOFT, 0);
            ui_button(CLAY_ID("Lod.Up"), "+", 34, PANEL, LINE_SOFT, 0);
        }
        CLAY(CLAY_ID("BottomBar.Meta"), {
            .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM, .childGap = 4 }
        }) {
            CLAY_TEXT(clay_string(state->slice_label), ui_text_config(TEXT, 15, CLAY_TEXT_WRAP_NONE));
            CLAY_TEXT(clay_string(state->lod_label), ui_text_config(DIM, 15, CLAY_TEXT_WRAP_NONE));
        }
        CLAY(CLAY_ID("BottomBar.Shortcuts"), {
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0) }, .layoutDirection = CLAY_TOP_TO_BOTTOM, .childGap = 4 }
        }) {
            CLAY_TEXT(clay_string("RMB orbit  WHEEL zoom  LMB pick  1-6 face  [ ] adjust  ` python"), ui_text_config(TEXT, 16, CLAY_TEXT_WRAP_WORDS));
            CLAY_TEXT(clay_string("TAB mode  X/Y/Z axis  PGUP/PGDN slice  +/- lod  R reset  G grid  H help"), ui_text_config(DIM, 14, CLAY_TEXT_WRAP_WORDS));
        }
    }
}

static void build_inspector(KubeUiState *state, const KubeUiFrameDesc *frame) {
    CLAY(CLAY_ID("Inspector"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_FIXED(KUBE_UI_INSPECTOR_W), .height = CLAY_SIZING_GROW(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM
        },
        .backgroundColor = clay_color(PANEL),
        .border = border_config(LINE_SOFT, 1)
    }) {
        CLAY(CLAY_ID("Inspector.Scroll"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                .padding = {16, 16, 14, 14},
                .childGap = 10,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .clip = { .horizontal = true, .vertical = true, .childOffset = Clay_GetScrollOffset() }
        }) {
            CLAY_TEXT(CLAY_STRING("CELL INSPECTOR"), ui_text_config(TEXT, 18, CLAY_TEXT_WRAP_NONE));
            if (!frame->editor_has_selection) {
                CLAY_TEXT(CLAY_STRING("Select a visible cell or slice cube to inspect live packed values."), ui_text_config(DIM, 16, CLAY_TEXT_WRAP_WORDS));
                CLAY_TEXT(CLAY_STRING("Right-drag keeps orbit control on the viewport."), ui_text_config(SUBTLE, 15, CLAY_TEXT_WRAP_WORDS));
            } else {
                CLAY_TEXT(clay_string(state->inspector_title), ui_text_config(TEXT, 18, CLAY_TEXT_WRAP_NONE));
                CLAY_TEXT(clay_string(state->inspector_packed), ui_text_config(DIM, 16, CLAY_TEXT_WRAP_NONE));
                CLAY_TEXT(clay_string(state->inspector_state), ui_text_config(frame->read_only ? TEXT : DIM, 15, CLAY_TEXT_WRAP_WORDS));
                for (int face_index = 0; face_index < 6; face_index++) {
                    int active = face_index == frame->sel_face;
                    Color outline = active ? INVERSE : LINE_FAINT;
                    Color fill = active ? PANEL_SOFT : PANEL_ALT;
                    CLAY(CLAY_IDI("Inspector.FaceRow", face_index), {
                        .layout = {
                            .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(36) },
                            .padding = {10, 10, 6, 6},
                            .childGap = 8,
                            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
                        },
                        .backgroundColor = clay_color(fill),
                        .cornerRadius = rounded_panel_radius(),
                        .border = border_config(outline, active ? 2 : 1)
                    }) {
                        CLAY(CLAY_IDI("Inspector.FaceText", face_index), {
                            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0) } }
                        }) {
                            CLAY_TEXT(clay_string(state->face_labels[face_index]), ui_text_config(TEXT, 16, CLAY_TEXT_WRAP_NONE));
                        }
                        ui_button(CLAY_IDI("Inspector.FaceMinus", face_index), "-", 34, PANEL, LINE_SOFT, 0);
                        ui_button(CLAY_IDI("Inspector.FacePlus", face_index), "+", 34, PANEL, LINE_SOFT, 0);
                    }
                }
                CLAY_TEXT(CLAY_STRING("STATE CHIPS"), ui_text_config(TEXT, 16, CLAY_TEXT_WRAP_NONE));
                if (frame->states <= 10) {
                    /* Compute chip width so all chips fit within the inspector panel.
                     * Available content width = panel width minus scroll padding (16+16).
                     * Each inter-chip gap is 6px; there are (states-1) gaps. */
                    static const float CHIP_GAP = 6.0f;
                    static const float CHIP_MAX_W = 28.0f;
                    float avail = (float)(KUBE_UI_INSPECTOR_W - 32);
                    float chip_w = (avail - (frame->states - 1) * CHIP_GAP) / (float)frame->states;
                    if (chip_w > CHIP_MAX_W) chip_w = CHIP_MAX_W;
                    if (chip_w < 16.0f) chip_w = 16.0f;
                    CLAY(CLAY_ID("Inspector.Chips"), {
                        .layout = { .childGap = (int)CHIP_GAP, .layoutDirection = CLAY_LEFT_TO_RIGHT }
                    }) {
                        for (int value = 0; value < frame->states; value++) {
                            ui_chip(CLAY_IDI("Inspector.Chip", value), state->chip_labels[value], color_for_state(value, frame->states), value == frame->selected_face_values[frame->sel_face], chip_w);
                        }
                    }
                } else {
                    CLAY_TEXT(CLAY_STRING("Use [ ] when the state range is larger than the chip rail."), ui_text_config(DIM, 14, CLAY_TEXT_WRAP_WORDS));
                }
                CLAY_TEXT(CLAY_STRING("VALUE SCALE"), ui_text_config(TEXT, 16, CLAY_TEXT_WRAP_NONE));
                CLAY(CLAY_ID("Inspector.Legend"), { .layout = { .childGap = 8 } }) {
                    for (int value = 0; value < (frame->states < 8 ? frame->states : 8); value++) {
                        Color fill = color_for_state(value, frame->states);
                        CLAY(CLAY_IDI("Inspector.LegendChip", value), {
                            .layout = { .sizing = { .width = CLAY_SIZING_FIXED(28), .height = CLAY_SIZING_FIXED(14) } },
                            .backgroundColor = clay_color(fill),
                            .cornerRadius = CLAY_CORNER_RADIUS(0),
                            .border = border_config(LINE_SOFT, 1)
                        }) {}
                    }
                }
            }
        }
    }
}

static void build_hover_card(KubeUiState *state, const KubeUiFrameDesc *frame) {
    if (!frame->hover_has_cell || frame->show_help || frame->show_console) return;
    CLAY(CLAY_ID("HoverCard"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_FIXED(214), .height = CLAY_SIZING_FIXED(86) },
            .padding = {12, 12, 10, 10},
            .childGap = 4,
            .layoutDirection = CLAY_TOP_TO_BOTTOM
        },
        .backgroundColor = clay_color(ColorAlpha(PANEL, 0.96f)),
        .cornerRadius = rounded_panel_radius(),
        .border = border_config(LINE, 1),
        .floating = {
            .offset = {18, 18},
            .parentId = CLAY_ID("Viewport").id,
            .zIndex = 20,
            .attachPoints = { .element = CLAY_ATTACH_POINT_LEFT_TOP, .parent = CLAY_ATTACH_POINT_LEFT_TOP },
            .attachTo = CLAY_ATTACH_TO_ELEMENT_WITH_ID
        }
    }) {
        CLAY_TEXT(CLAY_STRING("HOVER"), ui_text_config(DIM, 15, CLAY_TEXT_WRAP_NONE));
        CLAY_TEXT(clay_string(state->hover_line1), ui_text_config(TEXT, 17, CLAY_TEXT_WRAP_NONE));
        CLAY_TEXT(clay_string(state->hover_line2), ui_text_config(DIM, 15, CLAY_TEXT_WRAP_WORDS));
    }
}

static void build_help_overlay(void) {
    CLAY(CLAY_ID("HelpPanel"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_FIXED(KUBE_UI_HELP_W), .height = CLAY_SIZING_FIXED(KUBE_UI_HELP_H) },
            .padding = {20, 20, 18, 18},
            .childGap = 12,
            .layoutDirection = CLAY_TOP_TO_BOTTOM
        },
        .backgroundColor = clay_color(ColorAlpha(PANEL, 0.98f)),
        .cornerRadius = rounded_panel_radius(),
        .border = border_config(LINE, 1),
        .floating = {
            .zIndex = 40,
            .attachPoints = { .element = CLAY_ATTACH_POINT_CENTER_CENTER, .parent = CLAY_ATTACH_POINT_CENTER_CENTER },
            .attachTo = CLAY_ATTACH_TO_ROOT
        }
    }) {
        CLAY(CLAY_ID("HelpHeader"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(38) },
                .padding = {12, 12, 8, 8},
                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
            },
            .backgroundColor = clay_color(INVERSE),
            .border = border_config(INVERSE, 1)
        }) {
            CLAY_TEXT(CLAY_STRING("KEYMAP / HELP"), ui_text_config(INK, 20, CLAY_TEXT_WRAP_NONE));
        }
        CLAY_TEXT(CLAY_STRING("Wheel or PgUp/PgDn scrolls this panel. Esc closes it. Ctrl+Q exits the editor."), ui_text_config(DIM, 15, CLAY_TEXT_WRAP_WORDS));
        CLAY(CLAY_ID("HelpScroll"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                .childGap = 12,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .clip = { .horizontal = true, .vertical = true, .childOffset = Clay_GetScrollOffset() }
        }) {
            for (int section_index = 0; section_index < (int)(sizeof(HELP_SECTIONS) / sizeof(HELP_SECTIONS[0])); section_index++) {
                CLAY(CLAY_IDI("HelpSection", section_index), {
                    .layout = {
                        .sizing = { .width = CLAY_SIZING_GROW(0) },
                        .padding = {14, 14, 12, 12},
                        .childGap = 6,
                        .layoutDirection = CLAY_TOP_TO_BOTTOM
                    },
                    .backgroundColor = clay_color(ColorAlpha(PANEL_ALT, 0.92f)),
                    .cornerRadius = rounded_panel_radius(),
                    .border = border_config(section_index == 0 ? LINE : LINE_SOFT, 1)
                }) {
                    CLAY_TEXT(clay_string(HELP_SECTIONS[section_index].title), ui_text_config(TEXT, 16, CLAY_TEXT_WRAP_NONE));
                    CLAY_TEXT(clay_string(HELP_SECTIONS[section_index].body), ui_text_config(DIM, 15, CLAY_TEXT_WRAP_WORDS));
                }
            }
        }
    }
}

static void build_console_overlay(KubeUiState *state, const KubeUiFrameDesc *frame) {
    float panel_w = console_panel_width(frame->screen_w);
    float panel_h = console_panel_height(frame->screen_h);

    CLAY(CLAY_ID("ConsolePanel"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_FIXED(panel_w), .height = CLAY_SIZING_FIXED(panel_h) },
            .padding = {14, 14, 12, 12},
            .childGap = 10,
            .layoutDirection = CLAY_TOP_TO_BOTTOM
        },
        .backgroundColor = clay_color(ColorAlpha(PANEL, 0.98f)),
        .cornerRadius = rounded_panel_radius(),
        .border = border_config(LINE, 1),
        .floating = {
            .offset = {state->console_position.x, state->console_position.y},
            .zIndex = 50,
            .attachPoints = { .element = CLAY_ATTACH_POINT_LEFT_TOP, .parent = CLAY_ATTACH_POINT_LEFT_TOP },
            .attachTo = CLAY_ATTACH_TO_ROOT
        }
    }) {
        CLAY(CLAY_ID("ConsoleHeader"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(34) },
                .padding = {10, 10, 6, 6},
                .childGap = 8,
                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
            },
            .backgroundColor = clay_color(ColorAlpha(PANEL_ALT, 0.98f)),
            .cornerRadius = rounded_panel_radius(),
            .border = border_config(LINE_SOFT, 1)
        }) {
            CLAY(CLAY_ID("ConsoleHeader.Left"), {
                .layout = {
                    .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                    .childGap = 8,
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
                }
            }) {
                CLAY(CLAY_ID("ConsoleHeader.Tag"), {
                    .layout = {
                        .padding = {8, 8, 5, 5}
                    },
                    .backgroundColor = clay_color(TERM_BLUE),
                    .border = border_config(TERM_BLUE, 1)
                }) {
                    CLAY_TEXT(CLAY_STRING("PY"), ui_text_config(INK, 13, CLAY_TEXT_WRAP_NONE));
                }
                CLAY_TEXT(CLAY_STRING("PYTHON / DEBUG"), ui_text_config(TEXT, 17, CLAY_TEXT_WRAP_NONE));
            }
            CLAY(CLAY_ID("ConsoleClose"), {
                .layout = {
                    .sizing = { .width = CLAY_SIZING_FIXED(26), .height = CLAY_SIZING_FIXED(22) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                },
                .backgroundColor = clay_color(Clay_Hovered() ? TERM_RED : PANEL_SOFT),
                .border = border_config(Clay_Hovered() ? TERM_RED : LINE_SOFT, 1)
            }) {
                CLAY_TEXT(CLAY_STRING("X"), ui_text_config(Clay_Hovered() ? INK : TEXT, 14, CLAY_TEXT_WRAP_NONE));
            }
        }
        CLAY_TEXT(CLAY_STRING("Drag header to move. Tab completes. Enter runs. Wheel scrolls output. Esc closes console."), ui_text_config(DIM, 13, CLAY_TEXT_WRAP_WORDS));
        CLAY_TEXT(clay_string(state->console_context), ui_text_config(TERM_AMBER, 13, CLAY_TEXT_WRAP_WORDS));
        if (frame->console_hint != NULL && frame->console_hint[0] != '\0') {
            CLAY(CLAY_ID("ConsoleHint"), {
                .layout = {
                    .sizing = { .width = CLAY_SIZING_GROW(0) },
                    .padding = {10, 10, 8, 8}
                },
                .backgroundColor = clay_color(ColorAlpha(PANEL_SOFT, 0.96f)),
                .border = border_config(TERM_BLUE, 1)
            }) {
                CLAY_TEXT(clay_string(frame->console_hint), ui_text_config(TERM_BLUE, 13, CLAY_TEXT_WRAP_WORDS));
            }
        }
        CLAY(CLAY_ID("ConsoleOutputHeader"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIT(0) },
                .childGap = 8,
                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
            }
        }) {
            CLAY(CLAY_ID("ConsoleOutputLabel"), {
                .layout = { .padding = {8, 8, 5, 5} },
                .backgroundColor = clay_color(frame->console_last_ok ? TERM_GREEN : TERM_RED),
                .border = border_config(frame->console_last_ok ? TERM_GREEN : TERM_RED, 1)
            }) {
                CLAY_TEXT(CLAY_STRING("OUT"), ui_text_config(INK, 13, CLAY_TEXT_WRAP_NONE));
            }
            CLAY_TEXT(clay_string(frame->console_last_ok ? "execution completed" : "execution raised an error"), ui_text_config(DIM, 13, CLAY_TEXT_WRAP_WORDS));
        }
        CLAY(CLAY_ID("ConsoleOutput"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                .padding = {10, 10, 10, 10},
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .backgroundColor = clay_color(ColorAlpha(PANEL_ALT, 0.92f)),
            .cornerRadius = rounded_panel_radius(),
            .border = border_config(frame->console_last_ok ? TERM_GREEN : TERM_RED, 1)
        }) {
            CLAY(CLAY_ID("ConsoleOutputScroll"), {
                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) }, .layoutDirection = CLAY_TOP_TO_BOTTOM },
                .clip = { .horizontal = true, .vertical = true, .childOffset = Clay_GetScrollOffset() }
            }) {
                CLAY_TEXT(clay_string(frame->console_output != NULL && frame->console_output[0] != '\0' ? frame->console_output : "No output yet. Try kube.inspect() or selected_cell"), ui_text_config(TEXT, 14, CLAY_TEXT_WRAP_WORDS));
            }
        }
        CLAY(CLAY_ID("ConsoleInput"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(42) },
                .padding = {12, 12, 10, 8}
            },
            .backgroundColor = clay_color(ColorAlpha(PANEL_ALT, 0.96f)),
            .cornerRadius = rounded_panel_radius(),
            .border = border_config(TERM_GREEN, 1)
        }) {
            CLAY_TEXT(clay_string(state->console_prompt), ui_text_config(TERM_GREEN, 16, CLAY_TEXT_WRAP_WORDS));
        }
    }
}

static void build_root_ui(KubeUiState *state, const KubeUiFrameDesc *frame) {
    CLAY(CLAY_ID("Root"), {
        .layout = {
            .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM
        }
    }) {
        build_top_bar(state, frame);
        CLAY(CLAY_ID("Content"), {
            .layout = {
                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                .layoutDirection = CLAY_LEFT_TO_RIGHT
            }
        }) {
            CLAY(CLAY_ID("Viewport"), {
                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } }
            }) {}
            build_inspector(state, frame);
        }
        build_bottom_bar(state, frame);
        build_hover_card(state, frame);
        if (frame->show_help) build_help_overlay();
        if (frame->show_console) build_console_overlay(state, frame);
    }
}

static Rectangle rect_from_box(Clay_BoundingBox box) {
    return (Rectangle){box.x, box.y, box.width, box.height};
}

static void clamp_console_position(KubeUiState *state, int screen_w, int screen_h) {
    float margin = 18.0f;
    float panel_w = console_panel_width(screen_w);
    float panel_h = console_panel_height(screen_h);
    float max_x = (float)screen_w - panel_w - margin;
    float max_y = (float)screen_h - panel_h - margin;
    if (max_x < margin) max_x = margin;
    if (max_y < margin) max_y = margin;
    state->console_position.x = clampf_local(state->console_position.x, margin, max_x);
    state->console_position.y = clampf_local(state->console_position.y, margin, max_y);
}

static void format_frame_strings(KubeUiState *state, const KubeUiFrameDesc *frame) {
    char visible_faces[96];
    format_visible_faces(frame->visible_faces, visible_faces, sizeof(visible_faces));
    snprintf(
        state->top_line1,
        sizeof(state->top_line1),
        "%s  |  %dx%dx%d  |  S=%d  |  %s",
        frame->file_label != NULL ? frame->file_label : "kube",
        frame->W,
        frame->H,
        frame->D,
        frame->states,
        frame->storage_label != NULL ? frame->storage_label : "storage"
    );
    snprintf(
        state->top_line2,
        sizeof(state->top_line2),
        "Mode:%s  Slice:%c=%d  LOD:%d  Faces:%d  Grid:%s  %s  Visible:%s",
        frame->view_mode == VIEW_SHELL ? "shell" : "slice",
        frame->slice_axis == 0 ? 'X' : (frame->slice_axis == 1 ? 'Y' : 'Z'),
        frame->slice_index,
        frame->lod,
        frame->rendered_faces,
        frame->show_grid ? "on" : "off",
        frame->read_only ? "READ ONLY" : "AUTOSAVE",
        frame->view_mode == VIEW_SHELL ? visible_faces : "All"
    );
    snprintf(state->slice_label, sizeof(state->slice_label), "Slice %d", frame->slice_index);
    snprintf(state->lod_label, sizeof(state->lod_label), "LOD %d", frame->lod);
    if (frame->status_text != NULL) {
        strncpy(state->status_text, frame->status_text, sizeof(state->status_text) - 1);
        state->status_text[sizeof(state->status_text) - 1] = '\0';
    } else {
        state->status_text[0] = '\0';
    }
    if (frame->editor_has_selection) {
        static const char *face_names[6] = {"Top","Bottom","Left","Right","Front","Back"};
        snprintf(state->inspector_title, sizeof(state->inspector_title), "Cell (%d, %d, %d)", frame->sel_x, frame->sel_y, frame->sel_z);
        snprintf(state->inspector_packed, sizeof(state->inspector_packed), "Packed: %llu", (unsigned long long)frame->selected_packed);
        snprintf(state->inspector_state, sizeof(state->inspector_state), "Viewer state: %s", frame->read_only ? "READ ONLY" : "LIVE AUTOSAVE");
        for (int face_index = 0; face_index < 6; face_index++) {
            snprintf(state->face_labels[face_index], sizeof(state->face_labels[face_index]), "%d. %-6s  %d", face_index + 1, face_names[face_index], frame->selected_face_values[face_index]);
        }
    }
    if (frame->hover_has_cell) {
        static const char *face_names[6] = {"Top","Bottom","Left","Right","Front","Back"};
        snprintf(state->hover_line1, sizeof(state->hover_line1), "Cell (%d, %d, %d)", frame->hover_x, frame->hover_y, frame->hover_z);
        snprintf(state->hover_line2, sizeof(state->hover_line2), "%s face = %d", face_names[frame->hover_face], frame->hover_face_value);
    }
    if (frame->editor_has_selection) {
        static const char *face_names[6] = {"Top","Bottom","Left","Right","Front","Back"};
        snprintf(state->console_context, sizeof(state->console_context), "selection=(%d,%d,%d) face=%s", frame->sel_x, frame->sel_y, frame->sel_z, face_names[frame->sel_face]);
    } else {
        strncpy(state->console_context, "selection=None", sizeof(state->console_context) - 1);
        state->console_context[sizeof(state->console_context) - 1] = '\0';
    }
    if (frame->hover_has_cell) {
        static const char *face_names[6] = {"Top","Bottom","Left","Right","Front","Back"};
        char hover_text[96];
        snprintf(hover_text, sizeof(hover_text), "  hover=(%d,%d,%d) face=%s", frame->hover_x, frame->hover_y, frame->hover_z, face_names[frame->hover_face]);
        strncat(state->console_context, hover_text, sizeof(state->console_context) - strlen(state->console_context) - 1);
    } else {
        strncat(state->console_context, "  hover=None", sizeof(state->console_context) - strlen(state->console_context) - 1);
    }
    snprintf(state->console_prompt, sizeof(state->console_prompt), ">>> %s_", frame->console_input != NULL ? frame->console_input : "");
    for (int value = 0; value < 10; value++) snprintf(state->chip_labels[value], sizeof(state->chip_labels[value]), "%d", value);
}

KubeUiState *kube_ui_create(Font font, int screen_w, int screen_h) {
    KubeUiState *state = (KubeUiState *)calloc(1, sizeof(KubeUiState));
    Clay_ErrorHandler error_handler = {0};
    Clay_Arena arena;
    if (state == NULL) return NULL;
    state->font = font;
    state->arena_capacity = Clay_MinMemorySize();
    state->arena_memory = malloc(state->arena_capacity);
    if (state->arena_memory == NULL) {
        free(state);
        return NULL;
    }
    error_handler.errorHandlerFunction = kube_ui_error_handler;
    error_handler.userData = state;
    arena = Clay_CreateArenaWithCapacityAndMemory(state->arena_capacity, state->arena_memory);
    state->context = Clay_Initialize(arena, (Clay_Dimensions){(float)screen_w, (float)screen_h}, error_handler);
    Clay_SetCurrentContext(state->context);
    Clay_SetMeasureTextFunction(kube_ui_measure_text, state);
    state->viewport_rect = (Rectangle){0, (float)KUBE_UI_TOP_BAR_H, (float)(screen_w - KUBE_UI_INSPECTOR_W), (float)(screen_h - KUBE_UI_TOP_BAR_H - KUBE_UI_BOTTOM_BAR_H)};
    state->inspector_rect = (Rectangle){(float)(screen_w - KUBE_UI_INSPECTOR_W), (float)KUBE_UI_TOP_BAR_H, (float)KUBE_UI_INSPECTOR_W, (float)(screen_h - KUBE_UI_TOP_BAR_H - KUBE_UI_BOTTOM_BAR_H)};
    return state;
}

void kube_ui_destroy(KubeUiState *state) {
    if (state == NULL) return;
    free(state->arena_memory);
    free(state);
}

void kube_ui_update(KubeUiState *state, const KubeUiFrameDesc *frame, KubeUiActions *actions) {
    Clay_ElementData viewport_data;
    Clay_ElementData inspector_data;
    Clay_ElementData console_panel_data;
    Clay_ElementData console_header_data;
    Clay_ElementData console_close_data;
    Clay_ScrollContainerData scroll_data;

    if (state == NULL || frame == NULL || actions == NULL) return;
    memset(actions, 0, sizeof(*actions));
    actions->clicked_axis = -1;
    actions->clicked_face_row = -1;
    actions->clicked_face_adjust_row = -1;
    actions->clicked_chip_value = -1;

    Clay_SetCurrentContext(state->context);
    Clay_SetLayoutDimensions((Clay_Dimensions){(float)frame->screen_w, (float)frame->screen_h});
    format_frame_strings(state, frame);

    if (frame->show_console && !state->console_position_initialized) {
        state->console_position.x = ((float)frame->screen_w - console_panel_width(frame->screen_w)) * 0.5f;
        state->console_position.y = ((float)frame->screen_h - console_panel_height(frame->screen_h)) * 0.5f;
        state->console_position_initialized = 1;
    }
    clamp_console_position(state, frame->screen_w, frame->screen_h);

    if (!frame->show_console) state->console_dragging = 0;

    if (frame->page_up_pressed || frame->page_down_pressed) {
        if (frame->show_help) {
            scroll_data = Clay_GetScrollContainerData(CLAY_ID("HelpScroll"));
            if (scroll_data.found && scroll_data.scrollPosition != NULL) scroll_data.scrollPosition->y += frame->page_up_pressed ? 180.0f : -180.0f;
        }
        if (frame->show_console) {
            scroll_data = Clay_GetScrollContainerData(CLAY_ID("ConsoleOutputScroll"));
            if (scroll_data.found && scroll_data.scrollPosition != NULL) scroll_data.scrollPosition->y += frame->page_up_pressed ? 140.0f : -140.0f;
        }
    }

    console_panel_data = Clay_GetElementData(CLAY_ID("ConsolePanel"));
    console_header_data = Clay_GetElementData(CLAY_ID("ConsoleHeader"));
    console_close_data = Clay_GetElementData(CLAY_ID("ConsoleClose"));
    if (
        frame->show_console &&
        frame->mouse_left_pressed &&
        console_header_data.found &&
        console_panel_data.found &&
        point_in_box(frame->mouse, console_header_data.boundingBox) &&
        !(console_close_data.found && point_in_box(frame->mouse, console_close_data.boundingBox))
    ) {
        state->console_dragging = 1;
        state->console_drag_offset = (Vector2){frame->mouse.x - console_panel_data.boundingBox.x, frame->mouse.y - console_panel_data.boundingBox.y};
    }
    if (state->console_dragging) {
        if (frame->mouse_left_down) {
            state->console_position.x = frame->mouse.x - state->console_drag_offset.x;
            state->console_position.y = frame->mouse.y - state->console_drag_offset.y;
            clamp_console_position(state, frame->screen_w, frame->screen_h);
        } else {
            state->console_dragging = 0;
        }
    }

    Clay_SetPointerState((Clay_Vector2){frame->mouse.x, frame->mouse.y}, frame->mouse_left_down != 0);
    Clay_UpdateScrollContainers(true, (Clay_Vector2){0.0f, frame->wheel}, frame->delta_time);
    Clay_BeginLayout();
    build_root_ui(state, frame);
    state->commands = Clay_EndLayout();

    viewport_data = Clay_GetElementData(CLAY_ID("Viewport"));
    inspector_data = Clay_GetElementData(CLAY_ID("Inspector"));
    if (viewport_data.found) state->viewport_rect = rect_from_box(viewport_data.boundingBox);
    if (inspector_data.found) state->inspector_rect = rect_from_box(inspector_data.boundingBox);
    actions->viewport_rect = state->viewport_rect;
    actions->inspector_rect = state->inspector_rect;

    actions->pointer_over_ui =
        Clay_PointerOver(CLAY_ID("TopBar")) ||
        Clay_PointerOver(CLAY_ID("BottomBar")) ||
        Clay_PointerOver(CLAY_ID("Inspector")) ||
        (frame->show_help && Clay_PointerOver(CLAY_ID("HelpPanel"))) ||
        (frame->show_console && Clay_PointerOver(CLAY_ID("ConsolePanel")));

    if (frame->mouse_left_pressed) {
        if (frame->show_console && Clay_PointerOver(CLAY_ID("ConsoleClose"))) actions->clicked_console_close = 1;
        actions->clicked_shell = Clay_PointerOver(CLAY_ID("Mode.Shell"));
        actions->clicked_slice = Clay_PointerOver(CLAY_ID("Mode.Slice"));
        if (Clay_PointerOver(CLAY_ID("Axis.X"))) actions->clicked_axis = 0;
        else if (Clay_PointerOver(CLAY_ID("Axis.Y"))) actions->clicked_axis = 1;
        else if (Clay_PointerOver(CLAY_ID("Axis.Z"))) actions->clicked_axis = 2;
        if (Clay_PointerOver(CLAY_ID("Slice.Prev"))) actions->clicked_slice_delta = -1;
        else if (Clay_PointerOver(CLAY_ID("Slice.Next"))) actions->clicked_slice_delta = 1;
        if (Clay_PointerOver(CLAY_ID("Lod.Down"))) actions->clicked_lod_delta = 1;
        else if (Clay_PointerOver(CLAY_ID("Lod.Up"))) actions->clicked_lod_delta = -1;
        for (int face_index = 0; face_index < 6; face_index++) {
            if (Clay_PointerOver(CLAY_IDI("Inspector.FaceRow", face_index))) actions->clicked_face_row = face_index;
            if (Clay_PointerOver(CLAY_IDI("Inspector.FaceMinus", face_index))) {
                actions->clicked_face_adjust_row = face_index;
                actions->clicked_face_adjust_delta = -1;
            } else if (Clay_PointerOver(CLAY_IDI("Inspector.FacePlus", face_index))) {
                actions->clicked_face_adjust_row = face_index;
                actions->clicked_face_adjust_delta = 1;
            }
        }
        if (frame->states <= 10) {
            for (int value = 0; value < frame->states; value++) {
                if (Clay_PointerOver(CLAY_IDI("Inspector.Chip", value))) actions->clicked_chip_value = value;
            }
        }
    }

    state->last_show_console = frame->show_console;
}

static float clay_radius_to_roundness(Clay_CornerRadius radius, Rectangle rect) {
    float max_radius = radius.topLeft;
    float basis = rect.width < rect.height ? rect.width : rect.height;
    if (radius.topRight > max_radius) max_radius = radius.topRight;
    if (radius.bottomLeft > max_radius) max_radius = radius.bottomLeft;
    if (radius.bottomRight > max_radius) max_radius = radius.bottomRight;
    if (basis <= 0.0f) return 0.0f;
    return clampf_local(max_radius / (basis * 0.5f), 0.0f, 1.0f);
}

static void draw_clay_text(Font font, Clay_StringSlice slice, Clay_TextRenderData data, Vector2 position) {
    char stack_buffer[1024];
    char *buffer = stack_buffer;
    Color color = {(unsigned char)data.textColor.r, (unsigned char)data.textColor.g, (unsigned char)data.textColor.b, (unsigned char)data.textColor.a};
    if (slice.length < 0) return;
    if ((size_t)slice.length + 1 > sizeof(stack_buffer)) {
        buffer = (char *)malloc((size_t)slice.length + 1);
        if (buffer == NULL) return;
    }
    memcpy(buffer, slice.chars, (size_t)slice.length);
    buffer[slice.length] = '\0';
    DrawTextEx(font, buffer, position, (float)data.fontSize, (float)data.letterSpacing, color);
    if (buffer != stack_buffer) free(buffer);
}

void kube_ui_render(KubeUiState *state) {
    if (state == NULL) return;
    for (int index = 0; index < state->commands.length; index++) {
        Clay_RenderCommand *command = Clay_RenderCommandArray_Get(&state->commands, index);
        Rectangle rect;
        float roundness;
        if (command == NULL) continue;
        rect = (Rectangle){command->boundingBox.x, command->boundingBox.y, command->boundingBox.width, command->boundingBox.height};
        switch (command->commandType) {
            case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
                Color fill = {
                    (unsigned char)command->renderData.rectangle.backgroundColor.r,
                    (unsigned char)command->renderData.rectangle.backgroundColor.g,
                    (unsigned char)command->renderData.rectangle.backgroundColor.b,
                    (unsigned char)command->renderData.rectangle.backgroundColor.a
                };
                roundness = clay_radius_to_roundness(command->renderData.rectangle.cornerRadius, rect);
                if (roundness > 0.0f) DrawRectangleRounded(rect, roundness, 8, fill);
                else DrawRectangleRec(rect, fill);
            } break;
            case CLAY_RENDER_COMMAND_TYPE_BORDER: {
                Clay_BorderRenderData border = command->renderData.border;
                Color color = {(unsigned char)border.color.r, (unsigned char)border.color.g, (unsigned char)border.color.b, (unsigned char)border.color.a};
                roundness = clay_radius_to_roundness(border.cornerRadius, rect);
                if (border.width.left == border.width.right && border.width.left == border.width.top && border.width.left == border.width.bottom && border.width.left > 0) {
                    if (roundness > 0.0f) DrawRectangleRoundedLinesEx(rect, roundness, 8, (float)border.width.left, color);
                    else DrawRectangleLinesEx(rect, (float)border.width.left, color);
                } else {
                    if (border.width.top > 0) DrawRectangleRec((Rectangle){rect.x, rect.y, rect.width, (float)border.width.top}, color);
                    if (border.width.bottom > 0) DrawRectangleRec((Rectangle){rect.x, rect.y + rect.height - border.width.bottom, rect.width, (float)border.width.bottom}, color);
                    if (border.width.left > 0) DrawRectangleRec((Rectangle){rect.x, rect.y, (float)border.width.left, rect.height}, color);
                    if (border.width.right > 0) DrawRectangleRec((Rectangle){rect.x + rect.width - border.width.right, rect.y, (float)border.width.right, rect.height}, color);
                }
            } break;
            case CLAY_RENDER_COMMAND_TYPE_TEXT:
                draw_clay_text(state->font, command->renderData.text.stringContents, command->renderData.text, (Vector2){rect.x, rect.y});
                break;
            case CLAY_RENDER_COMMAND_TYPE_SCISSOR_START:
                BeginScissorMode((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height);
                break;
            case CLAY_RENDER_COMMAND_TYPE_SCISSOR_END:
                EndScissorMode();
                break;
            default:
                break;
        }
    }
}
