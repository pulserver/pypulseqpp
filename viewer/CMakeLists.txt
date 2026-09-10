cmake_minimum_required(VERSION 3.21)
project(pypulseqpp_seqeyes LANGUAGES CXX)

# SeqEyes, built from the pinned submodule in seqeyes/ as it stands. What
# differs from its own build is only where Qt comes from: nothing is deployed
# beside the program, which finds the Qt that PySide6-Essentials installs.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(SEQEYES ${CMAKE_CURRENT_SOURCE_DIR}/seqeyes)
if(NOT EXISTS ${SEQEYES}/src/main.cpp)
    message(FATAL_ERROR
        "viewer/seqeyes is empty. Run:\n"
        "    git submodule update --init --recursive")
endif()

# SeqEyes' own OpenGL finder, which leaves out a framework recent macOS SDKs
# no longer ship.
list(PREPEND CMAKE_MODULE_PATH ${SEQEYES}/cmake)

# Qt's CMake packages hand OpenGL to anything linking Gui, through this
# target, and both finders stand down when it already exists. SeqEyes calls no
# OpenGL, so on Linux it is defined empty: linked, the GL dispatch libraries
# would be copied into the wheel beside the system's copy that PySide6's Qt
# loads, and two of them in one process crash it at start-up.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(WrapOpenGL::WrapOpenGL INTERFACE IMPORTED GLOBAL)
endif()

find_package(Qt6 6.8 REQUIRED COMPONENTS Core Concurrent Gui Widgets PrintSupport)

# The version SeqEyes reports is read from git by its own script, at build
# time; a build from the source distribution reports "unknown".
set(VERSION_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/version_autogen.cpp)
set_source_files_properties(${VERSION_CPP} PROPERTIES GENERATED TRUE)
add_custom_target(seqeyes_version ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND ${CMAKE_COMMAND} -DOUT=${VERSION_CPP} -DSRC_DIR=${SEQEYES}
            -P ${SEQEYES}/cmake/GenVersion.cmake
    BYPRODUCTS ${VERSION_CPP})

# SeqEyes' own source list, file for file. Listed rather than globbed, so a
# submodule bump that adds a file is a change someone decides to build.
set(SOURCES
    ${SEQEYES}/src/main.cpp
    ${SEQEYES}/src/InteractionHandler.cpp
    ${SEQEYES}/src/LogManager.cpp
    ${SEQEYES}/src/mainwindow.cpp
    ${SEQEYES}/src/PulseqLoadTransaction.cpp
    ${SEQEYES}/src/PulseqLoadUiAdapter.cpp
    ${SEQEYES}/src/PulseqLoader.cpp
    ${SEQEYES}/src/PulseqOpenController.cpp
    ${SEQEYES}/src/SeriesBuilder.cpp
    ${SEQEYES}/src/KSpaceTrajectory.cpp
    ${SEQEYES}/src/M1Calculator.cpp
    ${SEQEYES}/src/PnsCalculator.cpp
    ${SEQEYES}/src/Settings.cpp
    ${SEQEYES}/src/SettingsDialog.cpp
    ${SEQEYES}/src/RuntimeContext.cpp
    ${SEQEYES}/src/TRManager.cpp
    ${SEQEYES}/src/WaveformDrawer.cpp
    ${SEQEYES}/src/ExtensionPlotter.cpp
    ${SEQEYES}/src/ExtensionLegendDialog.cpp
    ${SEQEYES}/src/LogTableDialog.cpp
    ${SEQEYES}/src/doublerangeslider.cpp
    ${SEQEYES}/src/ZoomManager.cpp
    ${SEQEYES}/src/AutomationRunner.cpp
    ${SEQEYES}/src/TrajectoryColormap.cpp
    ${SEQEYES}/src/version_info.cpp
    ${SEQEYES}/src/external/pulseq/ExternalSequence.cpp
    ${SEQEYES}/src/external/pulseq/v151/ExternalSequence.cpp
    ${SEQEYES}/src/external/pulseq/v151/md5.cpp
    ${SEQEYES}/src/external/qcustomplot/qcustomplot.cpp
    ${VERSION_CPP}
)

# Headers are listed so moc sees every Q_OBJECT, including the ones with no
# source file of the same name.
set(HEADERS
    ${SEQEYES}/src/InteractionHandler.h
    ${SEQEYES}/src/LogManager.h
    ${SEQEYES}/src/mainwindow.h
    ${SEQEYES}/src/seqeyes_version.h
    ${SEQEYES}/src/version_info.h
    ${SEQEYES}/src/IPulseqLoadUi.h
    ${SEQEYES}/src/LoadResult.h
    ${SEQEYES}/src/OpenResult.h
    ${SEQEYES}/src/PulseqLoadTransaction.h
    ${SEQEYES}/src/PulseqLoadUiAdapter.h
    ${SEQEYES}/src/PulseqLoader.h
    ${SEQEYES}/src/PulseqOpenController.h
    ${SEQEYES}/src/NumericLineEdit.h
    ${SEQEYES}/src/SeriesBuilder.h
    ${SEQEYES}/src/KSpaceTrajectory.h
    ${SEQEYES}/src/M1Calculator.h
    ${SEQEYES}/src/PnsCalculator.h
    ${SEQEYES}/src/PulseqLabelAnalyzer.h
    ${SEQEYES}/src/Settings.h
    ${SEQEYES}/src/SettingsDialog.h
    ${SEQEYES}/src/TRManager.h
    ${SEQEYES}/src/WaveformDrawer.h
    ${SEQEYES}/src/ExtensionPlotter.h
    ${SEQEYES}/src/ExtensionLegendDialog.h
    ${SEQEYES}/src/ExtensionStyleMap.h
    ${SEQEYES}/src/LogTableDialog.h
    ${SEQEYES}/src/doublerangeslider.h
    ${SEQEYES}/src/ZoomManager.h
    ${SEQEYES}/src/AutomationRunner.h
    ${SEQEYES}/src/TrajectoryColormap.h
    ${SEQEYES}/src/external/pulseq/ExternalSequence.h
    ${SEQEYES}/src/external/pulseq/v151/ExternalSequence.h
    ${SEQEYES}/src/external/pulseq/v151/md5.h
    ${SEQEYES}/src/external/qcustomplot/qcustomplot.h
)

add_executable(seqeyes
    ${SOURCES}
    ${HEADERS}
    ${SEQEYES}/src/mainwindow.ui
    ${SEQEYES}/resources/resources.qrc
)
add_dependencies(seqeyes seqeyes_version)

target_include_directories(seqeyes PRIVATE
    ${SEQEYES}/src
    ${SEQEYES}/src/external/pulseq
    ${SEQEYES}/src/external/qcustomplot
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_compile_definitions(seqeyes PRIVATE
    _SILENCE_CXX17_BIND1ST_DEPRECATION_WARNING
    QT_MESSAGELOGCONTEXT
)
target_compile_options(seqeyes PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_link_libraries(seqeyes PRIVATE
    Qt6::Core Qt6::Concurrent Qt6::Gui Qt6::Widgets Qt6::PrintSupport)

# A Qt module the build finds but the program never calls -- Concurrent -- is
# left out of what it needs at run time.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(seqeyes PRIVATE LINKER:--as-needed)
endif()

# Installed as <site-packages>/pypulseqpp_seqeyes/bin/seqeyes, so PySide6's Qt
# is two directories up. Windows has no run path; the Python launcher puts
# PySide6 on PATH instead.
if(APPLE)
    set_target_properties(seqeyes PROPERTIES
        INSTALL_RPATH "@executable_path/../../PySide6/Qt/lib")
elseif(UNIX)
    set_target_properties(seqeyes PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../PySide6/Qt/lib")
endif()

if(WIN32)
    # A window, not a console: started from Explorer or Python alike.
    set_target_properties(seqeyes PROPERTIES WIN32_EXECUTABLE TRUE)
    set(ICON_RC ${CMAKE_CURRENT_BINARY_DIR}/seqeyes_icon.rc)
    file(WRITE ${ICON_RC} "IDI_ICON1 ICON \"${SEQEYES}/resources/images/logo.ico\"\n")
    target_sources(seqeyes PRIVATE ${ICON_RC})
endif()

install(TARGETS seqeyes RUNTIME DESTINATION pypulseqpp_seqeyes/bin)
