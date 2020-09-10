package com.eszdman.photoncamera.api;

import android.hardware.camera2.CaptureRequest;
import android.os.Handler;
import android.view.Surface;

import java.util.List;

public interface ICamera {

    interface CameraEvents
    {
        void onCameraOpen();
        void onCameraClose();
    }

    void setCameraEventsListner(CameraEvents cameraEventsListner);
    void onResume();
    void onPause();

    String getId();
    CaptureRequest.Builder createCaptureRequest(int template);
    void createCaptureSession(List<Surface> var1, android.hardware.camera2.CameraCaptureSession.StateCallback var2, Handler var3);
    void openCamera(String id);
    void closeCamera();
}
