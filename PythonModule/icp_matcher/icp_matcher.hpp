#ifndef ICP_MATCHER_HPP
#define ICP_MATCHER_HPP

#include "Eigen\Dense"
#include <algorithm>
#include <vector>

std::vector<int> alternating_icp_matching(std::vector<std::vector<float>> predictions, std::vector<std::vector<float>> true_outputs);

#endif 