# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hyj/DenseVisualOdometry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyj/DenseVisualOdometry/build

# Include any dependencies generated for this target.
include CMakeFiles/imgprolib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imgprolib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imgprolib.dir/flags.make

CMakeFiles/imgprolib.dir/src/FrameData.cpp.o: CMakeFiles/imgprolib.dir/flags.make
CMakeFiles/imgprolib.dir/src/FrameData.cpp.o: ../src/FrameData.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hyj/DenseVisualOdometry/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/imgprolib.dir/src/FrameData.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/imgprolib.dir/src/FrameData.cpp.o -c /home/hyj/DenseVisualOdometry/src/FrameData.cpp

CMakeFiles/imgprolib.dir/src/FrameData.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imgprolib.dir/src/FrameData.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hyj/DenseVisualOdometry/src/FrameData.cpp > CMakeFiles/imgprolib.dir/src/FrameData.cpp.i

CMakeFiles/imgprolib.dir/src/FrameData.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imgprolib.dir/src/FrameData.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hyj/DenseVisualOdometry/src/FrameData.cpp -o CMakeFiles/imgprolib.dir/src/FrameData.cpp.s

CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.requires:
.PHONY : CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.requires

CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.provides: CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.requires
	$(MAKE) -f CMakeFiles/imgprolib.dir/build.make CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.provides.build
.PHONY : CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.provides

CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.provides.build: CMakeFiles/imgprolib.dir/src/FrameData.cpp.o

CMakeFiles/imgprolib.dir/src/viewer.cpp.o: CMakeFiles/imgprolib.dir/flags.make
CMakeFiles/imgprolib.dir/src/viewer.cpp.o: ../src/viewer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hyj/DenseVisualOdometry/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/imgprolib.dir/src/viewer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/imgprolib.dir/src/viewer.cpp.o -c /home/hyj/DenseVisualOdometry/src/viewer.cpp

CMakeFiles/imgprolib.dir/src/viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imgprolib.dir/src/viewer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hyj/DenseVisualOdometry/src/viewer.cpp > CMakeFiles/imgprolib.dir/src/viewer.cpp.i

CMakeFiles/imgprolib.dir/src/viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imgprolib.dir/src/viewer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hyj/DenseVisualOdometry/src/viewer.cpp -o CMakeFiles/imgprolib.dir/src/viewer.cpp.s

CMakeFiles/imgprolib.dir/src/viewer.cpp.o.requires:
.PHONY : CMakeFiles/imgprolib.dir/src/viewer.cpp.o.requires

CMakeFiles/imgprolib.dir/src/viewer.cpp.o.provides: CMakeFiles/imgprolib.dir/src/viewer.cpp.o.requires
	$(MAKE) -f CMakeFiles/imgprolib.dir/build.make CMakeFiles/imgprolib.dir/src/viewer.cpp.o.provides.build
.PHONY : CMakeFiles/imgprolib.dir/src/viewer.cpp.o.provides

CMakeFiles/imgprolib.dir/src/viewer.cpp.o.provides.build: CMakeFiles/imgprolib.dir/src/viewer.cpp.o

# Object files for target imgprolib
imgprolib_OBJECTS = \
"CMakeFiles/imgprolib.dir/src/FrameData.cpp.o" \
"CMakeFiles/imgprolib.dir/src/viewer.cpp.o"

# External object files for target imgprolib
imgprolib_EXTERNAL_OBJECTS =

../lib/libimgprolib.a: CMakeFiles/imgprolib.dir/src/FrameData.cpp.o
../lib/libimgprolib.a: CMakeFiles/imgprolib.dir/src/viewer.cpp.o
../lib/libimgprolib.a: CMakeFiles/imgprolib.dir/build.make
../lib/libimgprolib.a: CMakeFiles/imgprolib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../lib/libimgprolib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/imgprolib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imgprolib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imgprolib.dir/build: ../lib/libimgprolib.a
.PHONY : CMakeFiles/imgprolib.dir/build

CMakeFiles/imgprolib.dir/requires: CMakeFiles/imgprolib.dir/src/FrameData.cpp.o.requires
CMakeFiles/imgprolib.dir/requires: CMakeFiles/imgprolib.dir/src/viewer.cpp.o.requires
.PHONY : CMakeFiles/imgprolib.dir/requires

CMakeFiles/imgprolib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imgprolib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imgprolib.dir/clean

CMakeFiles/imgprolib.dir/depend:
	cd /home/hyj/DenseVisualOdometry/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyj/DenseVisualOdometry /home/hyj/DenseVisualOdometry /home/hyj/DenseVisualOdometry/build /home/hyj/DenseVisualOdometry/build /home/hyj/DenseVisualOdometry/build/CMakeFiles/imgprolib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imgprolib.dir/depend
