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
include CMakeFiles/dvocorelib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dvocorelib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dvocorelib.dir/flags.make

CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o: CMakeFiles/dvocorelib.dir/flags.make
CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o: ../src/CalWeight.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hyj/DenseVisualOdometry/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o -c /home/hyj/DenseVisualOdometry/src/CalWeight.cpp

CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hyj/DenseVisualOdometry/src/CalWeight.cpp > CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.i

CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hyj/DenseVisualOdometry/src/CalWeight.cpp -o CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.s

CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.requires:
.PHONY : CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.requires

CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.provides: CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.requires
	$(MAKE) -f CMakeFiles/dvocorelib.dir/build.make CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.provides.build
.PHONY : CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.provides

CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.provides.build: CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o

CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o: CMakeFiles/dvocorelib.dir/flags.make
CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o: ../src/TransformEstimate.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hyj/DenseVisualOdometry/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o -c /home/hyj/DenseVisualOdometry/src/TransformEstimate.cpp

CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hyj/DenseVisualOdometry/src/TransformEstimate.cpp > CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.i

CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hyj/DenseVisualOdometry/src/TransformEstimate.cpp -o CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.s

CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.requires:
.PHONY : CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.requires

CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.provides: CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.requires
	$(MAKE) -f CMakeFiles/dvocorelib.dir/build.make CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.provides.build
.PHONY : CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.provides

CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.provides.build: CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o

CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o: CMakeFiles/dvocorelib.dir/flags.make
CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o: ../src/Least_squares.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hyj/DenseVisualOdometry/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o -c /home/hyj/DenseVisualOdometry/src/Least_squares.cpp

CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hyj/DenseVisualOdometry/src/Least_squares.cpp > CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.i

CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hyj/DenseVisualOdometry/src/Least_squares.cpp -o CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.s

CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.requires:
.PHONY : CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.requires

CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.provides: CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.requires
	$(MAKE) -f CMakeFiles/dvocorelib.dir/build.make CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.provides.build
.PHONY : CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.provides

CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.provides.build: CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o

# Object files for target dvocorelib
dvocorelib_OBJECTS = \
"CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o" \
"CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o" \
"CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o"

# External object files for target dvocorelib
dvocorelib_EXTERNAL_OBJECTS =

../lib/libdvocorelib.a: CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o
../lib/libdvocorelib.a: CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o
../lib/libdvocorelib.a: CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o
../lib/libdvocorelib.a: CMakeFiles/dvocorelib.dir/build.make
../lib/libdvocorelib.a: CMakeFiles/dvocorelib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../lib/libdvocorelib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/dvocorelib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dvocorelib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dvocorelib.dir/build: ../lib/libdvocorelib.a
.PHONY : CMakeFiles/dvocorelib.dir/build

CMakeFiles/dvocorelib.dir/requires: CMakeFiles/dvocorelib.dir/src/CalWeight.cpp.o.requires
CMakeFiles/dvocorelib.dir/requires: CMakeFiles/dvocorelib.dir/src/TransformEstimate.cpp.o.requires
CMakeFiles/dvocorelib.dir/requires: CMakeFiles/dvocorelib.dir/src/Least_squares.cpp.o.requires
.PHONY : CMakeFiles/dvocorelib.dir/requires

CMakeFiles/dvocorelib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dvocorelib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dvocorelib.dir/clean

CMakeFiles/dvocorelib.dir/depend:
	cd /home/hyj/DenseVisualOdometry/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyj/DenseVisualOdometry /home/hyj/DenseVisualOdometry /home/hyj/DenseVisualOdometry/build /home/hyj/DenseVisualOdometry/build /home/hyj/DenseVisualOdometry/build/CMakeFiles/dvocorelib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dvocorelib.dir/depend
