# Model explanation
This page will be an explanation of the mathematical model that is implemented in this repo.

## Basis of the model
Cities are filled with data sources that provide information regarding the number of people in a given area. For some of these data sources, their primary goal is to count the number of people on the street (such as "Telraam"), for others this can be inferred by analyzing their internal algorithms. Because different sources collect data in totally different variants, the type of information they provide differs greatly. This affects the coverage of the data source, the frequency with which it collects data, whether it is snapshot information or cumulative information, whether it is all modalities or only pedestrians or motorized traffic,.... Nevertheless, it is obvious that by combining different data sources, it is possible to estimate the traffic density at street level and for different modalities. Merging these different data sources should increase accuracy. And if more data sources are available, it can be expected that the estimate will be more accurate. In this project, a model is developed and implemented to realize data fusion of geotemporal data from many different sources.

We start with a street plan and several data sources that provide counts, related to the number of people in a given area on this map. There are various data sources that provide information about a region and others that tell something about certain segments. 

Not all data sources provide information for all modalities. Some will be modality specific others will count groups of modalities together without distinction.

All these different data sources are converted into snapshot data: the (approximated) number of people in a certain area for a certain modality or group of modalities at a certain time. This information together with the known length of each street segment is directly translated into linear model constraints. This model is further extended with linear continuity constraints at each intersection (for each modality separately). In this way, it is ensured that the number of people leaving a street segment corresponds to the increase in population density in adjacent street segments. This approach also links the different time steps. Finally, the data fusion is realized by minimizing a cost function at each time step based on deviations from a (uniform) distribution. The end result is an estimate of the density for each modality and each street segment, as a function of time. Because all implemented constraints are linear in the density variables, while the cost function is quadratic in the density variables, the resulting data fusion model is a quadratic programming problem. Currently, OSQP is used to build and solve the model, using the python interface. 

## Input of the model
The following input is used for the data fusion model: 

1. `street_segments.csv` - A file that contains the identifiers, geometries, and lengths for all street segments, as well as the data sources and data source cells in which they reside. 
2. `intersection.csv` - A file with all intersections in the street network, with all the street segment ends connecting them. 
3. `modality_mapping.json` - A file with modalities metadata, such as maximum density and assumed speed 
4. `count_constraints_bounds.json` – A file that describes for each data source whether it is an upper and/or lower limit for the actual 
5. `all_data.csv` - Data file that contains the counts per time step and the modalities that the data sources consider. 

The first two files are the outcome of the roadcutting procedure. This is a procedure to retrieve this information from the road registery using the datafile with the count information. The code to do this is found in the following repostitory: [cityflows-road-cutter](https://github.com/imec-int/cityflows-road-cutter).

## Different types of data sources
As mentioned earlier, all data obtained from the data sources had to be converted into snapshot format. So when additional data sources become available, it should be possible to easily incorporate them into the data fusion model. That's why we've divided the data sources into different types, and each type has its own implementation. We have identified the following data source types: 

Snapshot data: A data source that provides counts of people who are present in a particular area at a certain point in time. They can be directly related to the street segment densities for each time step.

Cumulative non-unique counts at a point: A data source that represents the total number of people who have passed a given point in a given time frame. In order to be able to use this type of data in our data fusion model, it must also be converted into snapshot data. This is only possible if the (average) velocity v of the considered modality is also known, because this allows to obtain the average time that this modality spends in a street segment of length $L$ as $L/v$. Multiplying this average time spent in a segment by the counts per unit of time in that segment yields the total number of snapshots in the segment at time $t$. Therefore, the snapshot count $N_t$ can be obtained from the cumulative non-unique count $c_t$ for a time interval of $\mathcal{N}$ seconds in the following way:\
$N_t = \frac{c_tL}{\mathcal{N}v}$ 

For sources that count multiple modalities separately, this process will be completed separately for each of these modalities. An average speed is also used here, this is not a problem if the data source also provides this information. If not, the following assumption will be made here:
|**Modality**       | **Pedestrian**|**Bicycle**|**Car**   |
|-------------------|---------------|-----------|----------|
|**Reference speed**|5.4 km/h       |11.88 km/h |13.7 km/h |

## Projecting data sources onto the street network
Now that all data has been converted into snapshot format, they must provide information for the street plan. This is done by imposing the following conditions on the densities $ρ_{s,m,t}$ of the street segments $s$, modalities $m$, and time $t$, where $l_{s}$ is the length of street segments:\
$\sum_{s\in \mathfrak{A_{c}}}\left(l_{s}\sum_{m_{d}}\rho_{s,m_{d},t}\right) \leq \left(1+\alpha_{d,c,t}\right)N_{d,c,t}$  
$\sum_{s\in \mathfrak{A_{c}}}\left(l_{s}\sum_{m_{d}}\rho_{s,m_{d},t}\right) \geq \left(1-\alpha_{d,c,t}\right)N_{d,c,t}$


i.e. the sum of the counts of all segments which lie within the area $\mathfrak{A_{c}}$ of cell $c$ of data source $d$, summed over the modalities $m_d$ considered by data source $d$, should fall within an interval centered around the count $N_{d,c,t}$ for that data source cell. These constraints are imposed for all considered time steps $t$. The width of this interval is determined by the slack variables $\alpha_{d,c,t}$, which allow to relax these constraints a bit to account for problematic areas as well as inherent inaccuracies in the data.  Ideally, these slack variables should be as small as possible, which is why they are added to the cost function with a large weighting factor.\
These conditions remain mostly unchanged for sources that only conatin information of a single segemnt, only the sum over the segments will drop. Some sources are only an upper or a lower bound for the reality and thus these will only impose the corresponding condition.

## coupling of time steps
For each time step,  we construct the model and optimize it to obtain the densities based on the available data for that particular time step. However, we also couple subsequent timesteps. The main idea is here that we request continuity at each intersection: for every modality (except background, $bg$), the sum of the changes between different time steps in the number of people must be zero. This is done by imposing that the densities of the current time step can be obtainedby adding the change in density $\delta$ at the current time step to the calculated density of the previous time step for each street segment $s$:\
$\rho_{s,m,t+\Delta t} = \rho_{s,m,t} + \delta_{s,m,t,0} + \delta_{s,m,t,1}$\
$\rho_{s,bg,t+\Delta t} = \rho_{s,bg,t}$\
where 0 and 1 in the subscript of $\delta$ indicate the two ends of the street segments. The change in density is, in turn, determined by imposing that the sum of the changes in counts should be 0 for each intersection $\mathfrak{I}$:\
$\sum_{s\in\mathfrak{I}} l_{s}\delta_{s,m,t,e_{s,\mathfrak{I}}}=0$\
With $e_{s,\mathfrak{I}}$ the collection of ends of street $s$. To model the continuity as good as possible the timestap should be as small as possible. The choise has been made to use 5 minutes because that still makes sense with the measuring frequency of the sources used.\
For the first iteration this of course not possible because there is no previous timestep to compare to, hence these conditions will not be implemented for the first timestamp model.

## Modality mixing
To model the fact that people switch modalities, every 4th iteration modality mixing will be applied. This means that the coupling of the timesteps will not be for every modality seperate. Instead they will be valid for the entire population (sum of the modalities):\
$\sum_{m}\rho_{s,m,t+\Delta t} = \sum_{m}\rho_{s,m,t} + \delta_{s,t,0}+ \delta_{s,t,1}$\
$\sum_{s\in\mathfrak{I}} l_{s}\delta_{s,t,e_{s,\mathfrak{I}}}=0$

## Maximal density
Streets only have a finite capacity, ther will be imposed a maximal density. This will be done for every modality separate, the values used are in the following table:  
|**Modality**       |**background**|**Bicycle** |**Pedestrian**|**Motorised** |
|-------------------|--------------|------------|--------------|--------------|
|**Maximal density**|1 person/m    |2 person/m  |2 person/m    |0.556 person/m|


These values find their origin in the following reasoning:
- Background: 1/1m
- Bicycle: 2x1/1m
- Pedestrian: 2x1/1m
- Motorised: 2x1.39/5m

The average length of  a car is 5 meters, for the other modalities the length is asumed to be 1 meter.The factor 2 is because a street has 2 sides, and for the motorised modality the second factor of 1.39 is because that is the average number of people in a car.

## Objective function
The  objective  function,  or  cost  function,  determines  what  set  of  densities  is  the  best  (i.e.giving the lowest cost) within the constraints which are imposed in the model.

### Uniform distribution of the densities per data source cell:
For each data source cell the average density is calculated by dividing the total counts for that cell by the total street segment  length  within  that  cell.   The  objective  is  then  to  get  the  calculated  density for each street segment as close as possible to the average density of the data cell in which it is located. The cost which is minimized in this case is the sum of the quadratic differences between the calculated and the average density for each street segment, weighted with the length of the segment:\
$f_1\left(\rho_{s,m,t}\right) = \sum_{d,c} \sum_{s\in\mathfrak{A_{c}}}\left(l_s\sum_{m_d}\rho_{s,m_d,t}-l_s\frac{N_{d,c,t}}{\sum_{s\in\mathfrak{A_{c}}}l_s}\right)^2$\
Furthermore, the slack variables discussed in an earlier subsection also need to be included in the cost function. This is done by adding the following renormalization term to the cost function:\
$f_{\alpha} = \alpha_{w}\sum_{d,c}\alpha_{d,c,t}^{2}$\
with $\alpha_w = 10 000$ as the used weighting factor.

### Weighted distribution:
Because the uniform distribution gave unwanted artifacts in the results, it was replaced by a weighted distribution. The used weigths $w_{s,m}$, for segment $s$ and modality $m$, can be seen as the probalility to find a person of this modality in this street. They require a priori knowledge of the streetnetwork and therefore should be calculated with some care. The implemntation of a weighted distribution in the cost function is as follows:\
$f_2\left( \rho_{s,m,t}\right) = \sum_{d\in \mathcal{D}}\sum_{c\in \mathcal{C_d}}\sum_{s\in \mathcal{S_{d,c}}}\sum_{m\in \mathcal{M_d}} \left( l_{s,d,c}\rho_{s,m}-N_{d,c}\frac{l_{s,d,c}\omega_{s,m}}{\sum_{s\in\mathcal{S_{d,c}}}\sum_{m\in\mathcal{M_d}} l_{s,d,c}\omega_{s,m}}\right)^{2}$\
This approach also needs a default value for when there is no a priori knowledge of the streetnetwork. In this case all the weights would need to be put to 1. This then resembles the unifrom distribution again, but not quite. The weights csv-file will be needed to be created for every streetnetwork worked with, being with 1's or assumed weights. Such a distribution can be created using [this](https://github.com/imec-int/cityflows-ml-weights) repository.

# Summary of the possible iterations
|**Iteration**|model|
| --- | :---|
|**First timestep**<br><br><br><br><br>|**min** $\sum_{d\in \mathcal{D}}\sum_{c\in \mathcal{C_d}}\sum_{s\in \mathcal{S_{d,c}}}\sum_{m\in \mathcal{M_d}} \left( l_{s,d,c}\rho_{s,m}-N_{d,c}\frac{l_{s,d,c}\omega_{s,m}}{\sum_{s\in\mathcal{S_{d,c}}}\sum_{m\in\mathcal{M_d}} l_{s,d,c}\omega_{s,m}}\right)^{2}$<br>$ \hspace{0.5cm}+ \alpha_{w}\sum_{d,c}\alpha_{d,c,t}^{2}$<br>**s.t.:**$\sum_{s\in\mathfrak{A_{c}}} \left(l_s\sum_{m_d}\rho_{s,m_d,t}\right)\leq\left(1+\alpha_{d,c,t}\right)N_{d,c,t}$<br>$\hspace{0.5cm}\sum_{s\in\mathfrak{A_{c}}} \left(l_s\sum_{m_d}\rho_{s,m_d,t}\right)\geq\left(1-\alpha_{d,c,t}\right)N_{d,c,t}$<br>$\hspace{0.5cm}\rho_{s,m,t} \leq \rho_m^{max}$|
|**No modality<br> mixing** <br> <br> <br> <br> <br> <br> <br>|**min** $\sum_{d\in \mathcal{D}}\sum_{c\in \mathcal{C_d}}\sum_{s\in \mathcal{S_{d,c}}}\sum_{m\in \mathcal{M_d}} \left( l_{s,d,c}\rho_{s,m}-N_{d,c}\frac{l_{s,d,c}\omega_{s,m}}{\sum_{s\in\mathcal{S_{d,c}}}\sum_{m\in\mathcal{M_d}} l_{s,d,c}\omega_{s,m}}\right)^{2}$<br>$\hspace{0.5cm} + \alpha_{w}\sum_{d,c}\alpha_{d,c,t}^{2}$<br>**s.t.:**$\sum_{s\in\mathfrak{A_{c}}}\left(l_s\sum_{m_d}\rho_{s,m_d,t}\right)\leq\left(1+\alpha_{d,c,t}\right)N_{d,c,t}$<br>$\hspace{0.5cm}\sum_{s\in\mathfrak{A_{c}}} \left(l_s\sum_{m_d}\rho_{s,m_d,t}\right)\geq\left(1-\alpha_{d,c,t}\right)N_{d,c,t}$<br>$\hspace{0.5cm}\rho_{s,m,t} \leq \rho_m^{max}$<br>$\hspace{0.5cm}\rho_{s,m,t+\Delta t} = \rho_{s,m,t} + \delta_{s,m,t,0}+ \delta_{s,m,t,1}$<br>$\hspace{0.5cm}\rho_{s,bg,t+\Delta t} = \rho_{s,bg,t}$<br>$\hspace{0.5cm}\sum_{s\in\mathfrak{I}} l_{s}\delta_{s,m,t,e_{s,\mathfrak{I}}}=0$|
|**Modality<br> mixing** <br> <br> <br> <br> <br>|**min** $\sum_{d\in \mathcal{D}}\sum_{c\in \mathcal{C_d}}\sum_{s\in \mathcal{S_{d,c}}}\sum_{m\in \mathcal{M_d}} \left( l_{s,d,c}\rho_{s,m}-N_{d,c}\frac{l_{s,d,c}\omega_{s,m}}{\sum_{s\in\mathcal{S_{d,c}}}\sum_{m\in\mathcal{M_d}} l_{s,d,c}\omega_{s,m}}\right)^{2}$<br>$\hspace{0.5cm} + \alpha_{w}\sum_{d,c}\alpha_{d,c,t}^{2}$<br>**s.t.:**$\sum_{s\in\mathfrak{A_{c}}}\left(l_s\sum_{m_d}\rho_{s,m_d,t}\right)\leq\left(1+\alpha_{d,c,t}\right)N_{d,c,t}$<br>$\hspace{0.5cm}\sum_{s\in\mathfrak{A_{c}}} \left(l_s\sum_{m_d}\rho_{s,m_d,t}\right)\geq\left(1-\alpha_{d,c,t}\right)N_{d,c,t}$<br>$\hspace{0.5cm}\sum_m\rho_{s,m,t+\Delta t} = \sum_m\rho_{s,m,t} + \delta_{s,t,0}+ \delta_{s,t,1}$<br>$\hspace{.5cm}\sum_{s\in\mathfrak{I}} l_{s}\delta_{s,t,e_{s,\mathfrak{I}}}=0$|

This means that there are multiple solvers required, this was achieved by using classes. These classes are defined in `src/model/solve` and explained in [class_structure_explanation](./src/model/solve/class_structure_explanation.md).