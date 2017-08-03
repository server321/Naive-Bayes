#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class GNB {
public:
    
    vector<string> possible_labels = {"left","keep","right"};
    vector<string> features = {"s", "d", "s_dot", "d_dot"};
    int num_var = 4;
    
    /**
     * Constructor
     */
    GNB();
    
    /**
     * Destructor
     */
    virtual ~GNB();
    
    void train(vector<vector<double> > data, vector<string>  labels);
    
    string predict(vector<double>);
    double gaussian_prob(double obs, double mu, double sig);
    map<string, map<string, double>> means;
    map<string, map<string, double>> stds;
    
};

#endif
