cmake_minimum_required(VERSION 3.8)

project("GlNeuralNet")

file(GLOB_RECURSE SRC "src/cpp/*.cpp" "src/cpp/*.h")

# libraries

# glm
# H:\C++\_Libraries_\glm
# https://github.com/g-truc/glm
set(GLM_INCLUDE_DIR "H:/C++/_Libraries_/glm")

# https://www.glfw.org/
set(GLFW_INCLUDE_DIR "H:/C++/_Libraries_/glfw-3.4.bin.WIN64/include")
set(GLFW_LIBRARIES "H:/C++/_Libraries_/glfw-3.4.bin.WIN64/lib-vc2022/glfw3dll.lib")
set(GLFW_BIN "H:/C++/_Libraries_/glfw-3.4.bin.WIN64/lib-vc2022/glfw3.dll")

# H:\C++\_Libraries_\glew-2.1.0
# https://glew.sourceforge.net/
set(GLEW_INCLUDE_DIR "H:/C++/_Libraries_/glew-2.1.0/include")
set(GLEW_LIBRARIES "H:/C++/_Libraries_/glew-2.1.0/lib/Release/x64/glew32s.lib")
set(GLEW_BIN "H:/C++/_Libraries_/glew-2.1.0/bin/Release/x64/glew32.dll")

# https://github.com/nlohmann/json
set(JSON_INCLUDE_DIR "H:/C++/_Libraries_/json-include/include")

# https://github.com/opencv/opencv/releases/tag/4.11.0
set(${OPENCV_DIR} "H:/C++/_Libraries_/opencv_4.11.0/build")
find_package(OpenCV REQUIRED core imgcodecs REQUIRED PATHS H:/C++/_Libraries_/opencv_4.11.0/build NO_DEFAULT_PATH)

add_definitions(-DGLEW_STATIC)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")

# Add source to this project's executable.
add_executable(GlNeuralNet ${SRC})
#add_library (GlNeuralNet SHARED ${SRC})

include_directories(
        ${GLFW_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(GlNeuralNet
        PUBLIC
        ${GLFW_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${OpenCV_LIBS}

        # dxgi
        d3d11.lib
        dxguid.lib
        dxgi.lib
        opengl32.lib
)

install(TARGETS GlNeuralNet DESTINATION lib)
install(DIRECTORY src/cpp/ DESTINATION include FILES_MATCHING PATTERN "*.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET GlNeuralNet PROPERTY CXX_STANDARD 20)
endif ()

add_custom_command(TARGET GlNeuralNet POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GLFW_BIN}" "${GLEW_BIN}"
        $<TARGET_FILE_DIR:GlNeuralNet>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/src/shaders"
        $<TARGET_FILE_DIR:GlNeuralNet>/shaders
        COMMENT "Copying libs and shaders folder to output directory"
)
