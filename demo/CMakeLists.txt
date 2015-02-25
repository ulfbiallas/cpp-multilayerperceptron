cmake_minimum_required(VERSION 2.8)

project(MultilayerPerceptronDemo)

set(MAIN_SRCS main.cc ../MultilayerPerceptron.cc)


#OpenGL
find_package(OpenGL REQUIRED)


#GLUT
set(GLUT_INCLUDE_DIRS)
set(GLUT_LIBRARY glut64)


ADD_EXECUTABLE(main ${MAIN_SRCS})

include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} )
target_link_libraries(main ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )