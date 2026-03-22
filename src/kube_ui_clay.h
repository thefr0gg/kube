/*
 * kube_ui_clay.h - shared bridge types for the Clay-based native editor UI
 *
 * This header exists so `kube_renderer.c` can describe one UI frame in plain
 * C data, hand that state to the Clay layer, then receive a compact action
 * summary back without coupling the renderer loop to Clay internals.
 */

#ifndef KUBE_UI_CLAY_H
#define KUBE_UI_CLAY_H

#include "raylib.h"
#include <stdint.h>

typedef struct KubeUiState KubeUiState;

/* Snapshot of everything the UI layer needs to render one frame. */
typedef struct KubeUiFrameDesc {
    int screen_w;
    int screen_h;
    float delta_time;
    Vector2 mouse;
    float wheel;
    int mouse_left_down;
    int mouse_left_pressed;
    int page_up_pressed;
    int page_down_pressed;
    int show_help;
    int show_console;
    int show_grid;
    int console_last_ok;
    int view_mode;
    int slice_axis;
    int slice_index;
    int lod;
    int read_only;
    int W;
    int H;
    int D;
    int states;
    int rendered_faces;
    const int *visible_faces;
    const char *file_label;
    const char *storage_label;
    const char *status_text;
    int status_visible;
    int editor_has_selection;
    int sel_x;
    int sel_y;
    int sel_z;
    int sel_face;
    uint64_t selected_packed;
    const int *selected_face_values;
    int hover_has_cell;
    int hover_x;
    int hover_y;
    int hover_z;
    int hover_face;
    int hover_face_value;
    const char *console_input;
    const char *console_hint;
    const char *console_output;
} KubeUiFrameDesc;

/* High-level UI intents returned to the renderer after hit testing. */
typedef struct KubeUiActions {
    Rectangle viewport_rect;
    Rectangle inspector_rect;
    int pointer_over_ui;
    int clicked_console_close;
    int clicked_shell;
    int clicked_slice;
    int clicked_axis;
    int clicked_slice_delta;
    int clicked_lod_delta;
    int clicked_face_row;
    int clicked_face_adjust_row;
    int clicked_face_adjust_delta;
    int clicked_chip_value;
} KubeUiActions;

/* Lifecycle and per-frame entry points implemented in `kube_ui_clay.c`. */
KubeUiState *kube_ui_create(Font font, int screen_w, int screen_h);
void kube_ui_destroy(KubeUiState *state);
void kube_ui_update(KubeUiState *state, const KubeUiFrameDesc *frame, KubeUiActions *actions);
void kube_ui_render(KubeUiState *state);

#endif
