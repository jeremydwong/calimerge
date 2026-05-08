#pragma once
// StateManager.h — Thin QObject coordinator.
//
// Port of src/calimerge/gui/state.py StateManager.
//
// Responsibilities:
//   - Holds the current AppState value.
//   - Emits signals when state changes (stateChanged, camerasChanged, …).
//   - Provides update helpers that callers use instead of mutating AppState
//     fields directly.
//   - Does NOT contain business logic — that lives in pure functions or workers.
//
// Threading note: all methods must be called from the Qt main thread.
// Workers emit signals; StateManager slots receive them on the main thread.

#include "AppState.h"

#include <QObject>
#include <QString>

class StateManager : public QObject {
    Q_OBJECT

public:
    explicit StateManager(QObject *parent = nullptr);
    ~StateManager() override;

    // Read-only access to current state
    const AppState &state() const;

    // --- Update helpers ---
    // Each helper updates the relevant sub-struct and emits the appropriate signal.

    void updateStatus(const QString &message);
    void reportError(const QString &message);

    void setCurrentTab(int tab_index);
    void setPreviewing(bool is_previewing);

    void setRecording(const RecordingState &recording);
    void setProcessing(const ProcessingState &processing);

    // Camera map helpers
    void setCameraState(int port, const CameraState &cam_state);
    void removeCameraState(int port);
    void clearCameras();

signals:
    // Emitted on any AppState change
    void stateChanged(const AppState &state);

    // Focused signals for high-frequency or targeted consumers
    void camerasChanged();
    void recordingChanged(const RecordingState &recording);
    void calibrationChanged();
    void processingChanged(const ProcessingState &processing);

    // Status bar text
    void statusMessage(const QString &message);
    void errorOccurred(const QString &message);

public slots:
    void onUpdateStatus(const QString &message);

private:
    AppState m_state;
};
