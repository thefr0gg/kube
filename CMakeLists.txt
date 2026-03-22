cmake_minimum_required(VERSION 3.20)
project(kube_renderer LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(KUBE_USE_SYSTEM_RAYLIB "Prefer an installed raylib package before fetching it" ON)
set(KUBE_RAYLIB_ROOT "" CACHE PATH "Path to an unpacked raylib distribution containing include/ and lib/")

if(KUBE_RAYLIB_ROOT)
    if(NOT EXISTS "${KUBE_RAYLIB_ROOT}/include/raylib.h")
        message(FATAL_ERROR "KUBE_RAYLIB_ROOT must contain include/raylib.h: ${KUBE_RAYLIB_ROOT}")
    endif()

    find_library(KUBE_RAYLIB_LIBRARY
        NAMES raylib raylibdll libraylib libraylibdll
        PATHS "${KUBE_RAYLIB_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    if(NOT KUBE_RAYLIB_LIBRARY)
        message(FATAL_ERROR "KUBE_RAYLIB_ROOT must contain a raylib library inside lib/: ${KUBE_RAYLIB_ROOT}")
    endif()

    add_library(raylib UNKNOWN IMPORTED)
    set_target_properties(raylib PROPERTIES
        IMPORTED_LOCATION "${KUBE_RAYLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KUBE_RAYLIB_ROOT}/include"
    )

    if(WIN32)
        set_property(TARGET raylib APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "opengl32;gdi32;winmm"
        )
        find_file(KUBE_RAYLIB_DLL
            NAMES raylib.dll
            PATHS "${KUBE_RAYLIB_ROOT}" "${KUBE_RAYLIB_ROOT}/bin" "${KUBE_RAYLIB_ROOT}/lib"
            NO_DEFAULT_PATH
        )
    endif()
elseif(KUBE_USE_SYSTEM_RAYLIB)
    find_package(raylib 5.0 CONFIG QUIET)
endif()

if(NOT TARGET raylib)
    include(FetchContent)
    FetchContent_Declare(raylib
        # `GIT_SHALLOW TRUE` makes CMake clone raylib with `--depth 1`.
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.0
        GIT_SHALLOW    TRUE)

    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(raylib)
endif()

if(UNIX AND NOT APPLE)
    # raylib expects POSIX time/process APIs on Linux; strict C11 mode needs
    # explicit feature macros so timespec/nanosleep/readlink/clock_gettime exist.
    target_compile_definitions(raylib PRIVATE _POSIX_C_SOURCE=200809L _DEFAULT_SOURCE)
endif()

add_library(kube_renderer SHARED
    src/kube_renderer.c
    src/kube_ui_clay.c
)
target_include_directories(kube_renderer PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/third_party/clay"
)
target_link_libraries(kube_renderer PRIVATE raylib)

if(KUBE_RAYLIB_ROOT AND WIN32 AND KUBE_RAYLIB_DLL)
    add_custom_command(TARGET kube_renderer POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${KUBE_RAYLIB_DLL}"
            "$<TARGET_FILE_DIR:kube_renderer>/raylib.dll"
    )
endif()

if(WIN32)
    if(MINGW)
        target_link_options(kube_renderer PRIVATE -static-libgcc)
    endif()
endif()

set_target_properties(kube_renderer PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    OUTPUT_NAME "kube_renderer"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/kube"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/kube"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(kube_renderer PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_SOURCE_DIR}/kube"
        "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_SOURCE_DIR}/kube"
        "ARCHIVE_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}"
    )
endforeach()

add_library(kube_storage SHARED
    src/kube_storage.c
)

if(WIN32)
    if(MINGW)
        target_link_options(kube_storage PRIVATE -static-libgcc)
    endif()
endif()

set_target_properties(kube_storage PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    OUTPUT_NAME "kube_storage"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/kube"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/kube"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(kube_storage PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_SOURCE_DIR}/kube"
        "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_SOURCE_DIR}/kube"
        "ARCHIVE_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}"
    )
endforeach()

add_custom_target(kube_native
    DEPENDS kube_renderer kube_storage
)
