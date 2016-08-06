#include <vector>
#include <fstream>

#include "caffe/filler.hpp"
#include "caffe/layers/multi_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    void MultiOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        // automatically configured num_output
        // Added by David FAN Quan 2016.8.6
        //const int num_output = this->layer_param_.multi_output_param().num_output();
        layer_no = this->layer_param_.multi_output_param().layer_no();
        const string full_topo_file = this->layer_param_.multi_output_param().full_topo_file();
        std::ifstream _file(full_topo_file.c_str());
        int _t;
        int num_output = 0;
        for(int i = 0; i < 80; i++)     // fixed row number 80
        {
            for(int j = 0; j < 80; j++)     // fixed row number 80
            {
                _file >> _t;
                if(_t == layer_no)
                    num_output++;
            }
        }
        _file.close();

        bias_term_ = this->layer_param_.multi_output_param().bias_term();
        transpose_ = this->layer_param_.multi_output_param().transpose();
        N_ = num_output;
        const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.multi_output_param().axis());
        // Dimensions starting from "axis" are "flattened" into a single
        // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
        // and axis == 1, N inner products with dimension CHW are performed.
        K_ = bottom[0]->count(axis);
        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            if (bias_term_) {
                this->blobs_.resize(2);
            } else {
                this->blobs_.resize(1);
            }
            // Initialize the weights
            vector<int> weight_shape(2);
            if (transpose_) {
                weight_shape[0] = K_;
                weight_shape[1] = N_;
            } else {
                weight_shape[0] = N_;
                weight_shape[1] = K_;
            }
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            // fill the weights
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.multi_output_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());
            // If necessary, intiialize and fill the bias term
            if (bias_term_) {
                vector<int> bias_shape(1, N_);
                this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.multi_output_param().bias_filler()));
                    bias_filler->Fill(this->blobs_[1].get());
            }
        }  // parameter initialization
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

    template <typename Dtype>
    void MultiOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        // Figure out the dimensions
        const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.multi_output_param().axis());
        const int new_K = bottom[0]->count(axis);
        CHECK_EQ(K_, new_K)
        << "Input size incompatible with inner product parameters.";
        // The first "axis" dimensions are independent inner products; the total
        // number of these is M_, the product over these dimensions.
        M_ = bottom[0]->count(0, axis);
        // The top shape will be the bottom shape with the flattened axes dropped,
        // and replaced by a single axis with dimension num_output (N_).
        vector<int> top_shape = bottom[0]->shape();
        top_shape.resize(axis + 1);
        top_shape[axis] = N_;
        top[0]->Reshape(top_shape);
        // Set up the bias multiplier
        if (bias_term_) {
            vector<int> bias_shape(1, M_);
            bias_multiplier_.Reshape(bias_shape);
            caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
    }

    template <typename Dtype>
    void MultiOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const Dtype* weight = this->blobs_[0]->cpu_data();
        caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
            M_, N_, K_, (Dtype)1.,
            bottom_data, weight, (Dtype)0., top_data);
        if (bias_term_) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                bias_multiplier_.cpu_data(),
                this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
        }
    }

    template <typename Dtype>
    void MultiOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        if (this->param_propagate_down_[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            const Dtype* bottom_data = bottom[0]->cpu_data();
            // Gradient with respect to weight
            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                    K_, N_, M_,
                    (Dtype)1., bottom_data, top_diff,
                    (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
            } else {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                    N_, K_, M_,
                    (Dtype)1., top_diff, bottom_data,
                    (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
            }
        }
        if (bias_term_ && this->param_propagate_down_[1]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bias
            caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
            bias_multiplier_.cpu_data(), (Dtype)1.,
            this->blobs_[1]->mutable_cpu_diff());
        }
        if (propagate_down[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bottom data
            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                    M_, K_, N_,
                    (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                    (Dtype)0., bottom[0]->mutable_cpu_diff());
            } else {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                    M_, K_, N_,
                    (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                    (Dtype)0., bottom[0]->mutable_cpu_diff());
            }
        }
    }

    #ifdef CPU_ONLY
    STUB_GPU(MultiOutputLayer);
    #endif

    INSTANTIATE_CLASS(MultiOutputLayer);
    REGISTER_LAYER_CLASS(MultiOutput);
}   // namespace caffe
