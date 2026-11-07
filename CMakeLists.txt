cmake_minimum_required(VERSION 3.20)
project(kube_storage LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    DEPENDS kube_storage
)
