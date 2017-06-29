/*
 * particle_filter.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: Manuj Shinkar
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	std::default_random_engine gen;

	std::normal_distribution<double> N_x(x,std[0]);
	std::normal_distribution<double> N_y(y,std[1]);
	std::normal_distribution<double> N_theta(theta,std[2]);

	for(int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1);	
	}

	is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;

	for(int i = 0; i < particles.size(); i++) {

		double new_x;
		double new_y;
		double new_theta;

		if(yaw_rate == 0) {
			new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else {
			new_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}

		std::normal_distribution<double> N_x(new_x,std_pos[0]);
		std::normal_distribution<double> N_y(new_y,std_pos[1]);
		std::normal_distribution<double> N_theta(new_theta,std_pos[2]);		
	
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
			
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i = 0; i < observations.size(); i++)	{

		double min_distance = numeric_limits<double>::max();

		for(int j = 0; j < predicted.size(); j++)	{

			double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

			if(distance < min_distance)	{
				min_distance = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {

	    double p_x = particles[i].x;
	    double p_y = particles[i].y;
	    double p_theta = particles[i].theta;

	    // a vector to hold the map landmark locations predicted 
	    vector<LandmarkObs> predictions;

	    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

	    	LandmarkObs predLandmark;

			predLandmark.x = map_landmarks.landmark_list[j].x_f;
			predLandmark.y = map_landmarks.landmark_list[j].y_f;
			predLandmark.id = map_landmarks.landmark_list[j].id_i;

			predictions.push_back(predLandmark);

	    }

	    //space transformation from vehicle to map
		std::vector<LandmarkObs> trans_observations;
		LandmarkObs obs;

		for(int i = 0; i < observations.size(); i++)	{

			LandmarkObs trans_obs;
			obs = observations[i];

			trans_obs.x = cos(p_theta)*observations[i].x - sin(p_theta)*observations[i].y + p_x;
	     	trans_obs.y = sin(p_theta)*observations[i].x + cos(p_theta)*observations[i].y + p_y;
	      	trans_observations.push_back(trans_obs);
		}

	    // dataAssociation for the predictions and transformed observations on current particle
	    dataAssociation(predictions, trans_observations);

	    // reinit weight
	    particles[i].weight = 1.0;


	    // calculate particles final weight using multivariate gaussian probalility
	    for (int j = 0; j < trans_observations.size(); j++) {

			int association = trans_observations[j].id;

			double meas_x = trans_observations[j].x;
			double meas_y = trans_observations[j].y;

			double mu_x = p_x;
			double mu_y = p_y;

			for (unsigned int k = 0; k < predictions.size(); k++) {
		        if (predictions[k].id == association) {
		          mu_x = predictions[k].x;
		          mu_y = predictions[k].y;
		        }
		     }

			long double multipler = 1/(2*M_PI*std_landmark[0]*std_landmark[1])*exp(-(pow(meas_x-mu_x,2.0)/(2*pow(std_landmark[0],2.0))+pow(meas_y-mu_y,2.0)/(2*pow(std_landmark[1],2.0))));
			if(multipler > 0)	{
				particles[i].weight*= multipler;
				weights[i] = particles[i].weight;
			}

	    }
  	}
	
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::default_random_engine gen;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  std::vector<Particle> resample_particles;

  for(int i = 0; i < num_particles; i++)	{
  	resample_particles.push_back(particles[distribution(gen)]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
  

