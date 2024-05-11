package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import android.util.Log;

import com.particlesdevs.photoncamera.R;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.nodes.Node;
import com.particlesdevs.photoncamera.processing.render.NoiseModeler;

public class ESD3D extends Node {
    boolean needClose = false;
    public ESD3D(boolean closing) {
        super("", "ES3D");
        needClose = closing;
    }

    @Override
    public void Compile() {
    }

    @Override
    public void Run() {
        if(basePipeline.main4 == null)
            basePipeline.main4 = glUtils.medianDown(previousNode.WorkingTexture,4);
        GLTexture grad;

        if(previousNode.WorkingTexture != basePipeline.main3){
            grad = basePipeline.main3;
            WorkingTexture = basePipeline.getMain();
        }
        else {
            grad = basePipeline.getMain();
            WorkingTexture = basePipeline.main3;
        }
        //glUtils.ConvDiff(previousNode.WorkingTexture, grad, 0.f);




        {
            Log.d(Name, "NoiseS:" + basePipeline.noiseS + ", NoiseO:" + basePipeline.noiseO);
            glProg.setDefine("NOISES", basePipeline.noiseS);
            glProg.setDefine("NOISEO", basePipeline.noiseO);

            glProg.setDefine("INSIZE", basePipeline.mParameters.rawSize);
            float ks = 1.0f + Math.min((basePipeline.noiseS+basePipeline.noiseO) * 3.0f * 100000.f, 34.f);
            int msize = 5 + (int)ks - (int)ks%2;
            Log.d("ESD3D", "KernelSize: "+ks+" MSIZE: "+msize);
            glProg.setDefine("KERNELSIZE", ks);
            glProg.setDefine("MSIZE", msize);
            glProg.useAssetProgram("esd3d2");
            glProg.setTexture("NoiseMap", basePipeline.main4);
            glProg.setTexture("InputBuffer", previousNode.WorkingTexture);
            glProg.setTexture("GradBuffer", grad);
            glProg.drawBlocks(WorkingTexture);
        }
        glProg.closed = true;
        if(needClose) {
            basePipeline.main4.close();
            basePipeline.main4 = null;
        }
    }
}
