#include "icp_matcher.hpp"


std::vector<int> alternating_icp_matching(std::vector<std::vector<float>> predictions, std::vector<std::vector<float>> true_outputs)
{

	const int original_size = true_outputs.size();
	const int dim = true_outputs[0].size();

	std::vector<int> index(original_size, 0);

	std::vector<int> remaining_pred_indexes(original_size, 0);
	std::vector<int> remaining_true_output_indexes(original_size, 0);

	for (int i = 0; i < original_size; i++)
	{
		remaining_pred_indexes[i] = i;
		remaining_true_output_indexes[i] = i;
	}


	while (remaining_true_output_indexes.size() > 0) {

		int remaining_size = remaining_true_output_indexes.size();

		bool anchor_true = true;
		float urand = (float)rand() / (float)RAND_MAX;
		if (urand < 0.5f) {
			anchor_true = false;
		}



		int anchor_idx_idx = rand() % remaining_size;
		int anchor_idx = 0;
		float* anchor;
		if (anchor_true)
		{
			anchor_idx = remaining_true_output_indexes[anchor_idx_idx];
			remaining_true_output_indexes.erase(remaining_true_output_indexes.begin() + anchor_idx_idx);
			anchor = true_outputs[anchor_idx].data();
		}
		else {
			anchor_idx = remaining_pred_indexes[anchor_idx_idx];
			remaining_pred_indexes.erase(remaining_pred_indexes.begin() + anchor_idx_idx);
			anchor = predictions[anchor_idx].data();
		}
		Eigen::Map<Eigen::VectorXf> anchor_vect(anchor, dim);

		float best_match = std::numeric_limits<float>::infinity();
		int best_idx_idx = -1;


#pragma omp parallel for num_threads(8)
		for (int i = 0; i < remaining_size; ++i) {

			float* compare_data_ptr = nullptr;
			if (anchor_true)
			{
				compare_data_ptr = predictions[remaining_pred_indexes[i]].data();
			}
			else {
				compare_data_ptr = true_outputs[remaining_true_output_indexes[i]].data();
			}

			Eigen::Map<Eigen::VectorXf> compare_vect(compare_data_ptr, dim);

			float match = (compare_vect - anchor_vect).squaredNorm();

			if (match <= best_match) {
#pragma omp critical
				{
					if (match < best_match) {
						best_match = match;
						best_idx_idx = i;
					}
					
					if (match == best_match)
					{
						urand = (float)rand() / (float)RAND_MAX;
						if (urand < 0.5f) {
							best_match = match;
							best_idx_idx = i;
						}
					}

				}
			}

		}

		if (anchor_true)
		{
			int best_idx = remaining_pred_indexes[best_idx_idx];
			remaining_pred_indexes.erase(remaining_pred_indexes.begin() + best_idx_idx);
			index[anchor_idx] = best_idx;
		}
		else {
			int best_idx = remaining_true_output_indexes[best_idx_idx];
			remaining_true_output_indexes.erase(remaining_true_output_indexes.begin() + best_idx_idx);
			index[best_idx] = anchor_idx;
		}

	}


	return index;
}


/*

int main(void)
{


	//Test 1

	std::vector<std::vector<float>> set1;

	int dim = 10;
	int num_elements = 100;

	std::vector<float> tmp(dim, 0.0f);

	for (int i = 0; i < num_elements; i++)
	{
		for (float& num : tmp) {
			float urand = (float)rand() / (float)RAND_MAX;
			num = urand;
		}
		set1.push_back(tmp);
	}

	std::vector<std::vector<float>> set2 = set1;

	for (int i = 0; i < num_elements; i++)
	{
		int rand_idx = rand() % set2.size();
		set2[rand_idx].swap(set2[i]);
	}


	std::vector<int> index = alternating_icp_matching(set1, set2);


	for (int i = 1; i < index.size(); i++)
	{
		for (int j = 0; j < dim; j++)
		{
			float num1 = set2[i][j];
			float num2 = set1[index[i]][j];
			assert(num1 == num2);
		}
	}


	std::sort(index.begin(), index.end());
	for (int i = 1; i < index.size(); i++)
	{
		assert(index[i-1] + 1 == index[i]);
	}

}

*/