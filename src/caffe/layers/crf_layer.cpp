#include <algorithm>
#include <vector>
#include "caffe/layers/crf_layer.hpp"

namespace caffe
{
    template <typename Dtype>
    void CRFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        //TODO do nothing for now, just return the input blob

        //get cpu data
        const Dtype* bottom_data = bottom[0]->cpu_data();
        //get handle of output cpu data
        Dtype* top_data = top[0]->mutable_cpu_data();
        //get data count of bottom data
        const int count = bottom[0]->count();
        //TODO need some params?
        for(int i = 0; i < count; i++)
            top_data[i] = bottom_data[i];
    }

    template <typename Dtype>
    void CRFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        //TODO do nothing for now
        /*
        if(propagate_down[0])
        {
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            for(int i = 0; i < count; i++)
                bottom_diff[i] = 1;
        }
        */
    }

    #ifdef CPU_ONLY
    STUB_GPU(CRFLayer);
    #endif

    INSTANTIATE_CLASS(CRFLayer);
    REGISTER_LAYER_CLASS(CRF);  //I don't know if we need this
}   // namespace caffe
