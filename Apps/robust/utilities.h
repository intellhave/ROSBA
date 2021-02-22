#ifndef UTILITIES_H
#define UTILITIES_H
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

static void randomSampling(int N, int sampleSize, std::vector<int>& sampled_indices)
{
    if (sampleSize > N)
      sampleSize = N;

    std::vector<int> all_indices;
    for (int i = 0; i < N; ++i)
      all_indices.push_back(i);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);

    sampled_indices.clear();
    for (int i = 0; i < sampleSize; ++i){
      sampled_indices.push_back(all_indices[i]);
      /* sampled_indices.push_back(i); */
    }
}


static void randomSampling(int N, int sampleSize, std::vector<int>& sampled_indices, std::vector<int>& fixed_indices)
{
    if (sampleSize > N)
      sampleSize = N;

    std::vector<int> all_indices;
    for (int i = 0; i < N; ++i)
      all_indices.push_back(i);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);

    //Fill sampled subsets;
    sampled_indices.clear();
    for (int i = 0; i < sampleSize; ++i){
      sampled_indices.push_back(all_indices[i]);
      /* sampled_indices.push_back(i); */
    }
    
    //Fill fixed subsets;
    fixed_indices.clear();
    for (int i = sampleSize; i < N; ++i)
    {
        fixed_indices.push_back(all_indices[i]);  
    }
}




#endif
