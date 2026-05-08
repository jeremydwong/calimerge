// MainWindow.cpp — QMainWindow implementation.
//
// Skeleton only: creates four placeholder tabs and wires the status bar
// to StateManager::statusMessage. No real content yet.

#include "MainWindow.h"
#include "StateManager.h"

#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QStatusBar>

// ---------------------------------------------------------------------------
// Helper: create a bare placeholder widget for an unimplemented tab
// ---------------------------------------------------------------------------
static QWidget *make_placeholder_tab(const QString &label_text) {
    QWidget *w = new QWidget;
    QVBoxLayout *layout = new QVBoxLayout(w);
    QLabel *lbl = new QLabel(label_text, w);
    lbl->setAlignment(Qt::AlignCenter);
    layout->addWidget(lbl);
    return w;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
MainWindow::MainWindow(StateManager *state_manager, QWidget *parent)
    : QMainWindow(parent)
    , m_state_manager(state_manager)
    , m_tabs(nullptr)
    , m_status_label(nullptr)
{
    setWindowTitle("Calimerge");
    resize(1200, 800);

    setupMenuBar();
    setupTabs();
    setupStatusBar();

    // Connect StateManager status signal to our slot
    connect(m_state_manager, &StateManager::statusMessage,
            this, &MainWindow::onStatusMessage);
}

MainWindow::~MainWindow() = default;

// ---------------------------------------------------------------------------
// setupMenuBar
// ---------------------------------------------------------------------------
void MainWindow::setupMenuBar() {
    QMenuBar *mb = menuBar();

    QMenu *file_menu = mb->addMenu("&File");
    QAction *quit_action = file_menu->addAction("&Quit");
    connect(quit_action, &QAction::triggered, this, &MainWindow::close);

    QMenu *help_menu = mb->addMenu("&Help");
    Q_UNUSED(help_menu);
}

// ---------------------------------------------------------------------------
// setupTabs
// ---------------------------------------------------------------------------
void MainWindow::setupTabs() {
    m_tabs = new QTabWidget(this);
    m_tabs->addTab(make_placeholder_tab("Record tab — not yet implemented"),     "Record");
    m_tabs->addTab(make_placeholder_tab("Intrinsic tab — not yet implemented"),  "Intrinsic");
    m_tabs->addTab(make_placeholder_tab("Extrinsic tab — not yet implemented"),  "Extrinsic");
    m_tabs->addTab(make_placeholder_tab("Process tab — not yet implemented"),    "Process");
    setCentralWidget(m_tabs);
}

// ---------------------------------------------------------------------------
// setupStatusBar
// ---------------------------------------------------------------------------
void MainWindow::setupStatusBar() {
    m_status_label = new QLabel("Ready", this);
    statusBar()->addWidget(m_status_label, 1);
}

// ---------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------
void MainWindow::onStatusMessage(const QString &message) {
    if (m_status_label) {
        m_status_label->setText(message);
    }
}
