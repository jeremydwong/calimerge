#pragma once
// MainWindow.h — QMainWindow subclass for Calimerge.
//
// Four-tab layout:
//   Tab 0: Record     (cameras, preview, recording)
//   Tab 1: Intrinsic  (per-camera lens calibration)
//   Tab 2: Extrinsic  (multi-camera bundle adjustment)
//   Tab 3: Process    (tracking + triangulation)
//
// Nothing is functional yet — this is the scaffold that proves the build
// chain before any real GUI work is written.

#include <QMainWindow>
#include <QTabWidget>
#include <QStatusBar>
#include <QLabel>

// Forward declarations — avoids pulling full headers into every TU
class StateManager;
class QWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(StateManager *state_manager, QWidget *parent = nullptr);
    ~MainWindow() override;

public slots:
    void onStatusMessage(const QString &message);

private:
    void setupMenuBar();
    void setupTabs();
    void setupStatusBar();

    StateManager *m_state_manager;  // not owned
    QTabWidget   *m_tabs;
    QLabel       *m_status_label;
};
