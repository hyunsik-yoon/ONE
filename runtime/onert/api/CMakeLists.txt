file(GLOB_RECURSE API_SRC "*.cc")

set(ONERT_DEV nnfw-dev)
add_library(${ONERT_DEV} SHARED ${API_SRC})

# Public headers to publish
# nnfw_internal.h is header for runtime developer, so it will not be installed
# But runtime developer can use nnfw_internal.h by linking nnfw-dev
set(NNFW_API_HEADERS include/nnfw.h include/nnfw_experimental.h)

target_link_libraries(${ONERT_DEV} PUBLIC nnfw-nnapi-header)
target_link_libraries(${ONERT_DEV} PUBLIC onert_core)
target_link_libraries(${ONERT_DEV} PRIVATE jsoncpp tflite_loader circle_loader ${LIB_PTHREAD})
target_link_libraries(${ONERT_DEV} PRIVATE nnfw_common)
target_link_libraries(${ONERT_DEV} PRIVATE nnfw_coverage)

# find_library(LIBANDROID NAMES libandroid.so
#                         PATHS /home/eric/dev/one_build_android_ndk/ndk/platforms/android-27/arch-arm64/usr/lib
#                         REQUIRED)
# message(STATUS "${LIBANDROID}" )

find_library(LIBANDROID_LOG NAMES liblog.so
                        PATHS /home/eric/dev/one_build_android_ndk/ndk/platforms/android-27/arch-arm64/usr/lib
                        REQUIRED)
message(STATUS "${LIBANDROID_LOG}" )

target_link_libraries(${ONERT_DEV} PUBLIC ${LIBANDROID_LOG})

target_include_directories(${ONERT_DEV} PUBLIC include)
set_target_properties(${ONERT_DEV} PROPERTIES PUBLIC_HEADER "${NNFW_API_HEADERS}")

install(TARGETS ${ONERT_DEV}
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/nnfw)
