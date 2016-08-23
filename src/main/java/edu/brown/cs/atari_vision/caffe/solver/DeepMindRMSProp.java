package edu.brown.cs.atari_vision.caffe.solver;

import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by maroderi on 8/8/16.
 */
public class DeepMindRMSProp {

    float lr;

    float rms_decay;
    float min_gradient;

    FloatBlobVector tmpVec;
    FloatBlobVector gVec;
    FloatBlobVector g2Vec;

    FloatNet net;

    boolean gpu;

    public DeepMindRMSProp(FloatNet net) {
        net.blobs();
        new FloatBlobSharedVector();
    }

    public void step() {

    }

    public void updateGradient(int paramID) {
        FloatBlob g = gVec.get(paramID);
        FloatBlob g2 = g2Vec.get(paramID);
        FloatBlob tmp = tmpVec.get(paramID);
        FloatBlob blob = net.blobs().get(paramID);

        int N = g.count();

        if (gpu) {
            // self.g:mul(0.95):add(0.05, self.dw)
            caffe_cpu_axpby_float(N, 1-rms_decay, blob.gpu_diff(), rms_decay, g.mutable_gpu_diff());

            // self.tmp:cmul(self.dw, self.dw)
            // self.g2:mul(0.95):add(0.05, self.tmp)
            caffe_powx_float(N, blob.cpu_diff(), 2, tmp.mutable_cpu_diff());
            caffe_cpu_axpby_float(N, 1-rms_decay, tmp.cpu_diff(), rms_decay, g2.mutable_cpu_diff());

            // self.tmp:cmul(self.g, self.g)
            // self.tmp:mul(-1)
            // self.tmp:add(self.g2)
            // self.tmp:add(0.01)
            // self.tmp:sqrt()
            caffe_powx_float(N, g.cpu_diff(), 2, tmp.mutable_cpu_diff());
            caffe_sub_float(N, g2.cpu_diff(), tmp.cpu_diff(), tmp.mutable_cpu_diff());
            caffe_add_scalar_float(N, min_gradient, tmp.mutable_cpu_diff());
            caffe_powx_float(N, tmp.cpu_diff(), 0.5f, tmp.mutable_cpu_diff());

            // self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
            caffe_div_float(N, blob.cpu_diff(), tmp.cpu_diff(), blob.mutable_cpu_diff());
            caffe_scal_float(N, lr, blob.mutable_cpu_diff());
        } else {
            // self.g:mul(0.95):add(0.05, self.dw)
            caffe_cpu_axpby_float(N, 1-rms_decay, blob.cpu_diff(), rms_decay, g.mutable_cpu_diff());

            // self.tmp:cmul(self.dw, self.dw)
            // self.g2:mul(0.95):add(0.05, self.tmp)
            caffe_powx_float(N, blob.cpu_diff(), 2, tmp.mutable_cpu_diff());
            caffe_cpu_axpby_float(N, 1-rms_decay, tmp.cpu_diff(), rms_decay, g2.mutable_cpu_diff());

            // self.tmp:cmul(self.g, self.g)
            // self.tmp:mul(-1)
            // self.tmp:add(self.g2)
            // self.tmp:add(0.01)
            // self.tmp:sqrt()
            caffe_powx_float(N, g.cpu_diff(), 2, tmp.mutable_cpu_diff());
            caffe_sub_float(N, g2.cpu_diff(), tmp.cpu_diff(), tmp.mutable_cpu_diff());
            caffe_add_scalar_float(N, min_gradient, tmp.mutable_cpu_diff());
            caffe_powx_float(N, tmp.cpu_diff(), 0.5f, tmp.mutable_cpu_diff());

            // self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
            caffe_div_float(N, blob.cpu_diff(), tmp.cpu_diff(), blob.mutable_cpu_diff());
            caffe_scal_float(N, lr, blob.mutable_cpu_diff());
        }
    }
}
