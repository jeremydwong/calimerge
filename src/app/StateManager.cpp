// StateManager.cpp — thin coordinator implementation.
//
// Each update helper:
//   1. Mutates the relevant field(s) in m_state.
//   2. Emits the focused signal for that sub-state.
//   3. Emits stateChanged for any listener that watches the whole state.
//
// Skeleton only — workers, async ops, and business logic are Phase 3 work.

#include "StateManager.h"

#include <QString>

StateManager::StateManager(QObject *parent)
    : QObject(parent)
    , m_state(app_state_default())
{
}

StateManager::~StateManager() = default;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const AppState &StateManager::state() const {
    return m_state;
}

// ---------------------------------------------------------------------------
// Status / error
// ---------------------------------------------------------------------------

void StateManager::updateStatus(const QString &message) {
    QByteArray ba = message.toUtf8();
    strncpy(m_state.status_message, ba.constData(),
            sizeof(m_state.status_message) - 1);
    m_state.status_message[sizeof(m_state.status_message) - 1] = '\0';
    emit statusMessage(message);
    emit stateChanged(m_state);
}

void StateManager::reportError(const QString &message) {
    emit errorOccurred(message);
    updateStatus("Error: " + message);
}

// ---------------------------------------------------------------------------
// Tab / preview
// ---------------------------------------------------------------------------

void StateManager::setCurrentTab(int tab_index) {
    m_state.current_tab = tab_index;
    emit stateChanged(m_state);
}

void StateManager::setPreviewing(bool is_previewing) {
    m_state.is_previewing = is_previewing;
    emit stateChanged(m_state);
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

void StateManager::setRecording(const RecordingState &recording) {
    m_state.recording = recording;
    emit recordingChanged(recording);
    emit stateChanged(m_state);
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

void StateManager::setProcessing(const ProcessingState &processing) {
    m_state.processing = processing;
    emit processingChanged(processing);
    emit stateChanged(m_state);
}

// ---------------------------------------------------------------------------
// Camera map helpers
// ---------------------------------------------------------------------------

void StateManager::setCameraState(int port, const CameraState &cam_state) {
    m_state.cameras.insert(port, cam_state);
    emit camerasChanged();
    emit stateChanged(m_state);
}

void StateManager::removeCameraState(int port) {
    m_state.cameras.remove(port);
    emit camerasChanged();
    emit stateChanged(m_state);
}

void StateManager::clearCameras() {
    m_state.cameras.clear();
    emit camerasChanged();
    emit stateChanged(m_state);
}

// ---------------------------------------------------------------------------
// Slot alias (maps to updateStatus for signal/slot wiring convenience)
// ---------------------------------------------------------------------------

void StateManager::onUpdateStatus(const QString &message) {
    updateStatus(message);
}
