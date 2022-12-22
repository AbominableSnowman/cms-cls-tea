#include "../include/nlohmann/json.hpp"
using json = nlohmann::json;
#include <fstream>
#include <iostream>

//void test_function(){
	
//}

int main() {
    
	/*
	std::ifstream f("./testing_conditions.json");
	json conditions = json::parse(f);
	
	for(int experiment_n=1; experiment_n<=50; experiment_n++){
		
		
		int diffusion_on = conditions[std::to_string(experiment_n)]["diffusion_on"];
		int growth_on = conditions[std::to_string(experiment_n)]["growth_on"];
		int advection_on = conditions[std::to_string(experiment_n)]["advection_on"];
		int source_sink_cond = conditions[std::to_string(experiment_n)]["source_sink_condition"];
		double D = conditions[std::to_string(experiment_n)]["diffusion coefficient"]; 
		
		double velocity_magnitude = conditions[std::to_string(experiment_n)]["velocity"];
		double v_x = std::sqrt(velocity_magnitude);
		double v_y = std::sqrt(velocity_magnitude);
		double v[2] = {v_x,v_y};
		double t_max = conditions[std::to_string(experiment_n)]["t_max"];
		
		
		
		std::cout << std::boolalpha
		<< "diffusion_on: " << diffusion_on << "\n"
		<< "growth_on: " << growth_on << "\n"
		<< "advection_on: " << advection_on << "\n"
		<< "source_sink_cond: " << source_sink_cond << "\n"
		<< "D: " << D << "\n"
		<< "v_x: " << v_x << "\n"
		<< "v_y: " << v_y << "\n"
		<< "t_max: " << t_max << "\n";
		
		
		int max_iter = (int)(t_max / 0.0052342) + 1;
		std::cout << max_iter << std::endl;
	}
	
	
	std::ifstream f("./testing_conditions.json");
	json conditions = json::parse(f);
	
	for(int experiment_n=1; experiment_n<=10; experiment_n++){
		std::cout << "Experiment " << experiment_n << " finshed." << std::endl;
	}*/
	
	
	
    return 0;
}