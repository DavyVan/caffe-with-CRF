#ifndef MULTI_OUTPUT_LAYER_HPP_
#define MULTI_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
    template <typename Dtype>
    class MultiOutputLayer : public Layer<Dtype>
    {
        // The following code is copied from InnerProductLayer
    public:
        explicit MultiOutputLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "MultiOutput"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int M_;
        int K_;
        int N_;
        bool bias_term_;
        Blob<Dtype> bias_multiplier_;
        bool transpose_;  ///< if true, assume transposed weights

        //The following is added by David FAN Quan 2016.8.6
        int layer_no;   // but it seems like won't be used...except for LayerSetUp
    };
}   // namespace caffe

#endif  // MULTI_OUTPUT_LAYER_HPP_
