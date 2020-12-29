package com.eszdman.photoncamera.api;

import android.content.SharedPreferences;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.MeteringRectangle;
import android.hardware.camera2.params.TonemapCurve;
import android.util.Log;
import android.util.Range;

import com.eszdman.photoncamera.app.PhotonCamera;
import com.eszdman.photoncamera.control.TouchFocus;
import com.eszdman.photoncamera.processing.opengl.GLDrawParams;
import com.eszdman.photoncamera.settings.PreferenceKeys;
import com.eszdman.photoncamera.ui.camera.CameraFragment;

import static android.hardware.camera2.CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE;
import static android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION;
import static android.hardware.camera2.CaptureRequest.*;

public class Settings {
    public final int aeModeOn = CONTROL_AE_MODE_ON;
    private final String TAG = "Settings";
    //Preferences
    public int noiseReduction = NOISE_REDUCTION_MODE_OFF;
    public int frameCount;
    public int lumenCount;
    public int chromaCount;
    public boolean enhancedProcess;
    public boolean grid;
    public boolean watermark;
    public boolean energySaving;
    public boolean aFDebugData;
    public boolean roundEdge;
    public boolean align;
    public boolean hdrx;
    public boolean hdrxNR;
    public double saturation;
    public double sharpness;
    public double contrastMpy = 1.0;
    public int contrastConst = 0;//TODO
    public double noiseRstr;
    public double compressor;
    public double gain;
    public boolean rawSaver;
    public boolean QuadBayer;
    public int cfaPattern;
    public int theme;
    public boolean remosaic;//TODO
    public boolean eisPhoto;
    public boolean fpsPreview;
    public int alignAlgorithm;
    public String mCameraID;
    public CameraMode selectedMode;
    public float[] toneMap;
    public float[] gamma;
    //Other
    public boolean ManualMode = false;
    public String lastPicture = null;
    public int afMode = CONTROL_AF_MODE_CONTINUOUS_PICTURE;
    public MeteringRectangle[] initialAF = null;
    public MeteringRectangle[] initialAE = null;
    public int aeModeLock = CONTROL_AE_STATE_LOCKED;
    public int aeCurrentPrev = CONTROL_AE_MODE_ON;

    private SharedPreferences.Editor sharedPreferencesEditor;
    private SharedPreferences sharedPreferences;

    //==================================================================================================================
    public void setLastPicture(String lastPicture) {
        this.lastPicture = lastPicture;
    }

    public void loadCache() {
        noiseReduction = PreferenceKeys.isSystemNrOn();
        frameCount = PreferenceKeys.getFrameCountValue();
        align = PreferenceKeys.isDisableAligningOn();
        lumenCount = PreferenceKeys.getLumaNrValue();
        chromaCount = PreferenceKeys.getChromaNrValue();
        enhancedProcess = PreferenceKeys.isEnhancedProcessionOn();
        grid = PreferenceKeys.isShowGridOn();
        watermark = PreferenceKeys.isShowWatermarkOn();
        energySaving = PreferenceKeys.getBool(PreferenceKeys.Preference.KEY_ENERGY_SAVING);
        aFDebugData = PreferenceKeys.isAfDataOn();
        roundEdge = PreferenceKeys.isRoundEdgeOn();
        sharpness = PreferenceKeys.getSharpnessValue();
        contrastMpy = PreferenceKeys.getContrastValue();//TODO recheck
//        contrastConst = get(contrastConst, "ContrastConst");///////TODO
//        saturation = get(saturation, "Saturation");
        saturation = PreferenceKeys.getSaturationValue();
        compressor = PreferenceKeys.getCompressorValue();
        noiseRstr = PreferenceKeys.getFloat(PreferenceKeys.Preference.KEY_NOISESTR_SEEKBAR);
        gain = PreferenceKeys.getGainValue();
        hdrx = PreferenceKeys.isHdrxNrOn();
        cfaPattern = PreferenceKeys.getCFAValue();
        rawSaver = PreferenceKeys.isSaveRawOn();
        remosaic = PreferenceKeys.isRemosaicOn();
        eisPhoto = PreferenceKeys.isEisPhotoOn();
        QuadBayer = PreferenceKeys.isQuadBayerOn();
        fpsPreview = PreferenceKeys.isFpsPreviewOn();
        hdrxNR = PreferenceKeys.isHdrxNrOn();
        alignAlgorithm = PreferenceKeys.getAlignMethodValue();
        selectedMode = CameraMode.PHOTO;
        toneMap = parseToneMapArray();
        gamma = parseGammaArray();
        mCameraID = PreferenceKeys.getCameraID();
        theme = PreferenceKeys.getThemeValue();
    }

    public void saveID() {
        PreferenceKeys.setCameraID(mCameraID);
    }

    float[] parseToneMapArray() {
        String savedArrayAsString = PreferenceKeys.getToneMap();
        if (savedArrayAsString == null)
            return new float[0];
        String[] array = savedArrayAsString.replace("[", "").replace("]", "").split(",");
        float[] finalArray = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            finalArray[i] = Float.parseFloat(array[i].trim());
        }
        return finalArray;
    }
    float[] parseGammaArray() {
        String savedArrayAsString = PreferenceKeys.getPref(PreferenceKeys.Preference.GAMMA);
        if (savedArrayAsString == null)
            return new float[0];
        String[] array = savedArrayAsString.replace("[", "").replace("]", "").split(",");
        float[] finalArray = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            finalArray[i] = Float.parseFloat(array[i].trim());
        }
        return finalArray;
    }

    /*CameraMode getCameraMode() {
        switch (PreferenceKeys.getCameraMode()) {
            case (0):
                return CameraMode.PHOTO;
            case (1):
                return CameraMode.NIGHT;
            case (2):
                return CameraMode.UNLIMITED;
        }
        return CameraMode.PHOTO;
    }*/

    public void applyRes(CaptureRequest.Builder captureBuilder) {
        captureBuilder.set(CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
        if(PhotonCamera.getSettings().energySaving){
            GLDrawParams.TileSize = 8;
        } else {
            GLDrawParams.TileSize = 256;
        }
        //captureBuilder.set(CONTROL_AF_MODE, CONTROL_AF_MODE_OFF);
        //captureBuilder.set(STATISTICS_LENS_SHADING_MAP_MODE, STATISTICS_LENS_SHADING_MAP_MODE_ON);
        //captureBuilder.set(CONTROL_SCENE_MODE,CONTROL_SCENE_MODE_HDR);
        //captureBuilder.set(EDGE_MODE, EDGE_MODE_HIGH_QUALITY);
    }

    public void applyPrev(CaptureRequest.Builder captureBuilder) {
        Camera2ApiAutoFix.Apply();
        captureBuilder.set(CONTROL_AE_MODE, aeModeOn);
        //captureBuilder.set(COLOR_CORRECTION_MODE,COLOR_CORRECTION_MODE_HIGH_QUALITY);
        int[] stabilizationModes = CameraFragment.mCameraCharacteristics.get(LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
        if (stabilizationModes != null && stabilizationModes.length > 1) {
            captureBuilder.set(LENS_OPTICAL_STABILIZATION_MODE, LENS_OPTICAL_STABILIZATION_MODE_ON);//Fix ois bugs for preview and burst
        }
        //captureBuilder.set(CONTROL_AE_EXPOSURE_COMPENSATION,-1);
        Range<Integer> range = CameraFragment.mCameraCharacteristics.get(CONTROL_AE_COMPENSATION_RANGE);

        //if (selectedMode == CameraMode.NIGHT && range != null)
        //    captureBuilder.set(CONTROL_AE_EXPOSURE_COMPENSATION, (int) range.getUpper());

        /*Point size = new Point(Interface.getCameraFragment().mImageReaderPreview.getWidth(),Interface.getCameraFragment().mImageReaderPreview.getHeight());
        double sizex = size.x;
        double sizey = size.y;*/
        //captureBuilder.set(CONTROL_AE_TARGET_FPS_RANGE,new Range<>(24,60));
        /*MeteringRectangle[] rectm8 = new MeteringRectangle[2];
        rectm8[0] = new MeteringRectangle(new Point((int)(sizex/2.0),(int)(sizey/2.0)),new Size((int)(sizex*2.0/4.0),(int)(sizey*2.0/4.0)),10);
        rectm8[1] = new MeteringRectangle(new Point((int)(sizex/2.0),(int)(sizey/2.0)),new Size((int)(sizex/7),(int)(sizey/7)),30);
        MeteringRectangle[] rectaf = new MeteringRectangle[1];
        rectaf[0] =  new MeteringRectangle(new Point((int)(sizex/2.0),(int)(sizey/2.0)),new Size((int)(sizex/4),(int)(sizey/4)),10);
        //captureBuilder.set(CONTROL_AF_REGIONS,rectaf);
        captureBuilder.set(CONTROL_AE_REGIONS,rectm8);
        //captureBuilder.set(CONTROL_AF_MODE, Interface.getSettings().afMode);*/
        Object focus = captureBuilder.get(CONTROL_AF_MODE);
        Log.d(TAG, "InDeviceFocus:" + (int) (focus));
        if (focus != null) afMode = (int) focus;
        TouchFocus.onConfigured = false;
        initialAF = captureBuilder.get(CONTROL_AF_REGIONS);
        initialAE = captureBuilder.get(CONTROL_AE_REGIONS);
        //Interface.getTouchFocus().setFocus(size.x/2,size.y/2);
        TouchFocus.onConfigured = true;
        //captureBuilder.set(TONEMAP_MODE, TONEMAP_MODE_GAMMA_VALUE);
        /*float[] rgb = new float[64];
        for (int i = 0; i < 64; i += 2) {
            float x = ((float) i) / 64.f;
            rgb[i] = x;
            float output = 2.8114f * x + -3.5701f * x * x + 1.6807f * x * x * x;
            output = Math.max(output, 0.f);
            output = Math.min(output, 1.f);
            //Log.d(TAG,"Curve:"+output);
            rgb[i + 1] = output;
        }
        TonemapCurve tonemapCurve = new TonemapCurve(rgb, rgb, rgb);
        captureBuilder.set(TONEMAP_CURVE, tonemapCurve);*/

        /*float[] apertures = CameraFragment.mCameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES);
        if(apertures != null && apertures.length > 1){
            captureBuilder.set(LENS_APERTURE,apertures[1]);
        }*/
    }

    // =================================================================================================================
//TODO fix Import Export
    /*public void ExportSettings() {
//        save();
        Map<String, ?> allKeys = sharedPreferences.getAll();
        File configFile = new File(Environment.getExternalStorageDirectory() + "//DCIM//PhotonCamera//Settings.ini");
        Properties props = new Properties();
        try {
            if (!configFile.exists()) configFile.createNewFile();
            props.load(new FileInputStream(configFile));
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        for (Map.Entry<String, ?> entry : allKeys.entrySet()) {
            props.setProperty(entry.getKey(), entry.getValue().toString());
            Log.v(TAG, "setProperty:" + entry.getKey() + " = " + entry.getValue().toString());
        }

        try {
            props.store(new FileOutputStream(configFile), "PhotonCamera settings file");
        } catch (IOException e) {
            e.printStackTrace();
        }
        final Activity activity = PhotonCamera.getCameraActivity();
        if (activity != null) {
            activity.runOnUiThread(() -> Toast.makeText(activity, "Exported", Toast.LENGTH_SHORT).show());
        }
    }

    public String[] getArr(Properties props, String propname, String def) {
        String values = props.getProperty(propname, def);
        values = values.replace("]", "").replace("[", "").replace(" ", "");
        String outarr[] = values.split(",");
        return outarr;
    }

    public void ImportSettings() {
        Map<String, ?> allKeys = sharedPreferences.getAll();
        File configFile = new File(Environment.getExternalStorageDirectory() + "//DCIM//PhotonCamera//Settings.ini");
        Properties props = new Properties();
        try {
            if (!configFile.exists()) configFile.createNewFile();
            props.load(new FileInputStream(configFile));
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        mCameraID = props.getProperty("Camera");
        CameraManager2.cameraManager2.mCameraIDs.clear();
        for (String id : getArr(props, "Cameras", CameraManager2.cameraManager2.getCameraIdList().toString())) {
            Log.d(TAG, "CameraID:" + id);
            CameraManager2.cameraManager2.mCameraIDs.add(id);
        }
        Log.v(TAG, "Tonemap:" + props.getProperty("Tonemap"));
        sharedPreferencesEditor.putString("Tonemap", props.getProperty("Tonemap"));
        for (Map.Entry<Object, Object> entry : props.entrySet()) {
            String property = entry.getValue().toString();
            String name = entry.getKey().toString();
            if (name.charAt(5) == '_' && name.length() > 6) {
                Log.d(TAG, "Importing:" + name.charAt(6) + " " + name);
                switch (name.charAt(6)) {
                    case 'b':
                        sharedPreferencesEditor.putBoolean(name, Boolean.parseBoolean(property));
                        break;
                    case 'i':
                        sharedPreferencesEditor.putInt(name, Integer.parseInt(property));
                        break;
                    case 'd':
                        sharedPreferencesEditor.putFloat(name, Float.parseFloat(property));
                        break;
                    case 's':
                        sharedPreferencesEditor.putString(name, property);
                        break;
                    case 'm':
                        sharedPreferencesEditor.putInt(name, Integer.parseInt(property));
                        break;
                }
            }
        }
        sharedPreferencesEditor.apply();
        loadCache();
//        PhotonCamera.getSettingsActivity().set();
        final Activity activity = PhotonCamera.getCameraActivity();
        if (activity != null) {
            activity.runOnUiThread(() -> Toast.makeText(activity, "Imported", Toast.LENGTH_SHORT).show());
        }
    }*/
}
