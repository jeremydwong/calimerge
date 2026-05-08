// main.cpp — Qt application entry point for Calimerge.
//
// Creates QApplication, StateManager (coordinator), and MainWindow.
// Sets application metadata so QSettings and QStandardPaths use the
// right namespaced directories on every platform.

#include "StateManager.h"
#include "MainWindow.h"

#include <QApplication>
#include <QString>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    app.setApplicationName("Calimerge");
    app.setOrganizationName("Calimerge");
    app.setOrganizationDomain("calimerge.app");
    app.setApplicationVersion("0.1.0");

    StateManager state_manager;
    MainWindow   window(&state_manager);

    window.show();

    return app.exec();
}
