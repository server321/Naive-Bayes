#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.hpp"

#include <numeric>
#include <algorithm>

/**
 * Initializes GNB
 */
GNB::GNB() {
    
}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
    
    /*
     Trains the classifier with N data points and labels.
     
     INPUTS
     data - array of N observations
     - Each observation is a tuple with 4 values: s, d,
     s_dot and d_dot.
     - Example : [
     [3.5, 0.1, 5.9, -0.02],
     [8.0, -0.3, 3.0, 2.2],
     ...
		  	]
     
     labels - array of N labels
     - Each label is one of "left", "keep", or "right".
     */
    
    unsigned long N = labels.size(); // number of observations
    
    // First map class, then map features
    map<string, map<string, vector<double>>> totals_by_label;
    
    for (auto label: possible_labels) {
        for (auto f: features) {
            totals_by_label[label][f] = {};
        }
    }
    
    for (int i=0; i < N; ++i){
        for (int j = 0; j<num_var; ++j) {
            if (labels[i] == "left")
                totals_by_label["left"][features[j]].push_back(data[i][j]);
            else if (labels[i] == "keep")
                totals_by_label["keep"][features[j]].push_back(data[i][j]);
            if (labels[i] == "right")
                totals_by_label["right"][features[j]].push_back(data[i][j]);
        }
    }
    
    double sum, mean, sq_sum, stdev;
    for (auto label: possible_labels){
        for (auto f: features){
            vector<double> diff(totals_by_label[label][f].size());
            sum = std::accumulate(totals_by_label[label][f].begin(), totals_by_label[label][f].end(), 0.0);
            mean = sum / totals_by_label[label][f].size();
            means[label][f] = mean;
            std::transform(totals_by_label[label][f].begin(), totals_by_label[label][f].end(), diff.begin(), [mean](double x) { return x - mean; });
            sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            stdev = sqrt(sq_sum / (totals_by_label[label][f].size()-1));
            stds[label][f] = stdev;
            cout << label << " " << f << ": mean: " << mean << " standard deviaction: " << pow(stdev, 2);
            cout << endl;
        }
        cout << endl;
    }
    
}

double GNB::gaussian_prob(double obs, double mu, double sig){
    double num = pow((obs - mu), 2);
    double denum = 2*pow(sig, 2);
    double norm = 1 / sqrt(2*M_PI*pow(sig, 2));
    return norm * exp(-num/denum);
}

string GNB::predict(vector<double> sample)
{
    /*
     Once trained, this method is called and expected to return
     a predicted behavior for the given observation.
     
     INPUTS
     
     observation - a 4 tuple with s, d, s_dot, d_dot.
     - Example: [3.5, 0.1, 8.5, -0.2]
     
     OUTPUT
     
     A label representing the best guess of the classifier. Can
     be one of "left", "keep" or "right".
     """
     # TODO - complete this
     */
    map<string, double> probs;
    map<string, double>::iterator it;
    double sum = 0;
    
    for (auto label: possible_labels) {
        double product = 1.0;
        for (int i = 0; i < num_var; i++) {
            double likelihood = gaussian_prob(sample[i], means[label][features[i]], stds[label][features[i]]);
            product *= likelihood;
        }
        sum += product;
        probs[label] = product;
    }
    // normalize each probability
    for (it=probs.begin(); it!=probs.end(); ++it) {
        it->second = it->second / sum;
    }
    double p_best = 0;
    string l; // store the label of highest pro so far
    for (it=probs.begin(); it!=probs.end(); ++it) {
        if (it->second > p_best){
            l = it->first;
            p_best = it->second;
        }
    }
    return l;
    
    //	return this->possible_labels[1];
    
}
