package com.eszdman.photoncamera.processing.opengl.postpipeline;

import android.util.Log;

import com.eszdman.photoncamera.R;
import com.eszdman.photoncamera.app.PhotonCamera;
import com.eszdman.photoncamera.processing.opengl.GLFormat;
import com.eszdman.photoncamera.processing.opengl.GLTexture;
import com.eszdman.photoncamera.processing.opengl.nodes.Node;
import com.eszdman.photoncamera.processing.render.Parameters;

import java.nio.FloatBuffer;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_LINEAR;

public class Demosaic extends Node {
    public Demosaic() {
        super(0, "Demosaic");
    }

    @Override
    public void Compile() {}

    @Override
    public void Run() {
        PostPipeline postPipeline = (PostPipeline) (basePipeline);
        GLTexture glTexture;
        glTexture = previousNode.WorkingTexture;
        glProg.useProgram(R.raw.demosaicp1);
        glProg.setTexture("RawBuffer", glTexture);
        glProg.setVar("CfaPattern", basePipeline.mParameters.cfaPattern);
        if(PhotonCamera.getSettings().cfaPattern == -2) glProg.setDefine("QUAD","1");

        //GLTexture green = new GLTexture(params.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16));
        glProg.drawBlocks(basePipeline.main1);
        //Green Channel guided denoising
        GLTexture outp = previousNode.WorkingTexture;

        glProg.useProgram(R.raw.denoisebygreen);
        glProg.setTexture("RawBuffer",previousNode.WorkingTexture);
        glProg.setTexture("GreenBuffer",basePipeline.main1);
        //glProg.setVar("CfaPattern", params.cfaPattern);
        GLTexture prev = previousNode.WorkingTexture;
        outp = basePipeline.main2;
        glProg.drawBlocks(outp);

        /*glProg.useProgram(R.raw.medianfilterhotpixel);
        GLTexture t = prev;
        prev = outp;
        outp = t;
        glProg.setTexture("RawBuffer",prev);
        glProg.setVar("CfaPattern", params.cfaPattern);
        glProg.drawBlocks(outp);*/

        /*glProg.useProgram(R.raw.denoisebygreen);
        t = prev;
        prev = outp;
        outp = t;
        glProg.setTexture("RawBuffer",prev);
        glProg.setTexture("GreenBuffer",basePipeline.main1);
        glProg.drawBlocks(outp);*/
        glProg.useProgram(R.raw.demosaicp2);
        GLTexture GainMapTex = new GLTexture(basePipeline.mParameters.mapSize, new GLFormat(GLFormat.DataType.FLOAT_16,4),
                FloatBuffer.wrap(basePipeline.mParameters.gainMap),GL_LINEAR,GL_CLAMP_TO_EDGE);
        glProg.setTexture("RawBuffer", outp);
        glProg.setTexture("GreenBuffer", basePipeline.main1);
        glProg.setVar("whitePoint",basePipeline.mParameters.whitePoint);
        glProg.setTexture("GainMap",GainMapTex);
        glProg.setVar("CfaPattern", basePipeline.mParameters.cfaPattern);
        glProg.setVar("RawSize",basePipeline.mParameters.rawSize);
        for(int i =0; i<4;i++){
            basePipeline.mParameters.blackLevel[i]/=basePipeline.mParameters.whiteLevel*postPipeline.regenerationSense;
        }
        glProg.setVar("blackLevel",basePipeline.mParameters.blackLevel);
        WorkingTexture = basePipeline.main3;
        glProg.drawBlocks(WorkingTexture);
        GainMapTex.close();
        glProg.close();
    }
}
