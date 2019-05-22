#include "RefineDetPlugin.hpp"
#include "MathFunction.hpp"

namespace Shadow{

__global__ void applyConf_gpu(int batchSize, int _numPriorboxes, int _numClasses, float _objectness_score, const float *arm_conf, const float *odm_conf, float *conf, cudaStream_t stream)
{
    int priorboxesId = threadIdx.x + blockIdx.x * blockDim.x;
    if (priorboxesId < batchSize * _numPriorboxes)
    {
        if (arm_conf[2 * priorboxesId + 1] < _objectness_score)
        {
            for (int c = 0; c < _numClasses; ++c)
            {
                if (c != 0)
                    conf[priorboxesId * _numClasses + c] = 0.0;
                else
                    conf[priorboxesId * _numClasses + c] = 1.0;
            }
        }
        else
        {
            for (int c = 0; c < _numClasses; c++)
                conf[priorboxesId * _numClasses + c] = odm_conf[priorboxesId * _numClasses + c];
        }
    }
}

__global__ void applyLoc_gpu(int batchSize, int _numPriorboxes, const float *arm_loc, const float *priorbox_loc, float *loc)
{
    int box_id = threadIdx.x + blockIdx.x * blockDim.x;
    int beginAddress = box_id / _numPriorboxes * _numPriorboxes * 8;
    int box_id_image = box_id % _numPriorboxes;
    if (box_id < batchSize * _numPriorboxes)
    {

        // float xmin = priorbox_loc[beginAddress + box_id_image * 4],
        //       ymin = priorbox_loc[beginAddress + box_id_image * 4 + 1],
        //       xmax = priorbox_loc[beginAddress + box_id_image * 4 + 2],
        //       ymax = priorbox_loc[beginAddress + box_id_image * 4 + 3];
        // float var1 = priorbox_loc[beginAddress + (box_id_image + _numPriorboxes) * 4],
        //       var2 = priorbox_loc[beginAddress + (box_id_image + _numPriorboxes) * 4 + 1],
        //       var3 = priorbox_loc[beginAddress + (box_id_image + _numPriorboxes) * 4 + 2],
        //       var4 = priorbox_loc[beginAddress + (box_id_image + _numPriorboxes) * 4 + 3];

        float xmin = priorbox_loc[box_id_image * 4],
              ymin = priorbox_loc[box_id_image * 4 + 1],
              xmax = priorbox_loc[box_id_image * 4 + 2],
              ymax = priorbox_loc[box_id_image * 4 + 3];
        float var1 = priorbox_loc[(box_id_image + _numPriorboxes) * 4],
              var2 = priorbox_loc[(box_id_image + _numPriorboxes) * 4 + 1],
              var3 = priorbox_loc[(box_id_image + _numPriorboxes) * 4 + 2],
              var4 = priorbox_loc[(box_id_image + _numPriorboxes) * 4 + 3];


        float bbox1 = arm_loc[box_id * 4],
              bbox2 = arm_loc[box_id * 4 + 1],
              bbox3 = arm_loc[box_id * 4 + 2],
              bbox4 = arm_loc[box_id * 4 + 3];

        // if(xmin == priorbox_loc[1200] && ymin == priorbox_loc[1201] && xmax == priorbox_loc[1202] && ymax == priorbox_loc[1203]){
        //       printf("%d %d %d\n",box_id, _numPriorboxes, box_id_image);
        // }
        

        float prior_width = xmax - xmin,
              prior_height = ymax - ymin,
              prior_center_x = (xmax + xmin) / 2,
              prior_center_y = (ymax + ymin) / 2;
        float decode_bbox_center_x = var1 * bbox1 * prior_width + prior_center_x,
              decode_bbox_center_y = var2 * bbox2 * prior_height + prior_center_y,
              decode_bbox_width = exp(var3 * bbox3) * prior_width,
              decode_bbox_height = exp(var4 * bbox4) * prior_height;

        loc[beginAddress + box_id_image * 4] = decode_bbox_center_x - decode_bbox_width / 2;
        loc[beginAddress + box_id_image * 4 + 1] = decode_bbox_center_y - decode_bbox_height / 2;
        loc[beginAddress + box_id_image * 4 + 2] = decode_bbox_center_x + decode_bbox_width / 2;
        loc[beginAddress + box_id_image * 4 + 3] = decode_bbox_center_y + decode_bbox_height / 2;
        loc[beginAddress + (box_id_image + _numPriorboxes) * 4] = var1;
        loc[beginAddress + (box_id_image + _numPriorboxes) * 4 + 1] = var2;
        loc[beginAddress + (box_id_image + _numPriorboxes) * 4 + 2] = var3;
        loc[beginAddress + (box_id_image + _numPriorboxes) * 4 + 3] = var4;
    }
}

void applyConf(int batchSize, int _numPriorboxes, int _numClasses, float _objectness_score, const float *arm_conf, const float *odm_conf, float *conf, cudaStream_t stream)
{
    int block = GET_BLOCKS(batchSize * _numPriorboxes);
    int grid = (batchSize * _numPriorboxes + block - 1) / block;
    applyConf_gpu<<<grid, block>>>(batchSize, _numPriorboxes, _numClasses, _objectness_score, arm_conf, odm_conf, conf, stream);
}

void applyLoc(int batchSize, int _numPriorboxes, const float *arm_loc, const float *priorbox_loc, float *loc)
{
    int block = GET_BLOCKS(batchSize * _numPriorboxes);
    int grid = (batchSize * _numPriorboxes + block - 1) / block;
    applyLoc_gpu<<<grid, block>>>(batchSize, _numPriorboxes, arm_loc, priorbox_loc, loc);
}

}
