EIGEN_INCLUDE_DIR = $$PWD/../../3rdparty/eigen-5.0.0

INCLUDEPATH += $$EIGEN_INCLUDE_DIR

TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp
