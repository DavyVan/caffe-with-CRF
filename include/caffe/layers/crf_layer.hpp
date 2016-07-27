#ifndef CAFFE_CRF_LAYER_HPP_
#define CAFFE_CRF_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe
{
    template <typename Dtype>
    class CRFLayer : public NeuronLayer<Dtype>
    {
    public:
        explicit CRFLayer(const LayerParameter& param)
            :NeuronLayer<Dtype>(param) {}

        virtual inline const char* type() const {return "CRF";}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    };
}

#endif //CAFFE_CRF_LAYER_HPP_
