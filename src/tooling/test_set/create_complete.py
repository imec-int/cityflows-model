filename_streets = 'street_segments.csv'
filename_intersections = 'intersections.csv'
filename_data = 'counts/all_data_small_cropland_velo.csv'
foldername_data = 'data/managed_data_files/AAA/input'
foldername_writing = 'data/bigger_test_set/'

base_data_source = 'proximus' # has to be proximus at the moment, the files don't match cropland vs proximus

# from os.path import join
from matplotlib import colors
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os import path, pardir, mkdir
import shapely
import time
import numpy as np

def create_test_set(E,N,visual=False,writing=True):
    """ Creates a set of street_segments and intersection inside two base_data_source locationranges.
            INPUT: 
                listed coordinates of points in the location range of the following form
                    N N[i]° E E[i]°
                Boolean for optional visualization
                
                uses the data in cityflows-data-model/data/ready_for_main

            OUTPUT:
                writes 'selected_intersections.csv' and 'selected_street_segments.csv' 
                in the following folder cityflows-data-model/data/selected_test_set
    """
    if type(N) == float:
        N = [N]
    if type(E) == float:
        E = [E]
    if len(N) != len(E):
        raise Exception("The coordinates don't match in length")
    if len(E) < 1 :
        raise Exception("The list of the coordinates should be at least 1 long")


    THIS_DIR = path.dirname(__file__)
    data_files_path = path.join(THIS_DIR, pardir, pardir, pardir, foldername_data)
    if path.exists(data_files_path)==False:
        raise Exception("Directory:"+data_files_path+" doesn't exist.")

    #STEP1: load and transform (filter) data
    ## loading data
    street_path = path.join(data_files_path, filename_streets)
    intersections_path = path.join(data_files_path, filename_intersections)
    data_path = path.join(data_files_path, filename_data)

    df_street_segments_unfiltered = pd.read_csv(street_path)
    df_street_segments_unfiltered.loc[df_street_segments_unfiltered['data_source']=='cropland','data_source'] = 'proximus'
    df_intersections = pd.read_csv(intersections_path)
    df_data_unfiltered = pd.read_csv(data_path)
    print('DONE reading data')

    ## changing the column_name index to data_source_index if needed
    changed_index = False
    if "index" in df_data_unfiltered:
        df_data_unfiltered.rename({'index': 'data_source_index'},axis=1,inplace=True)
        changed_index = True
        print('Had to change the column_name index to data_source_index')

    ## filtering on duplicates
    df_data = df_data_unfiltered.copy()
    df_data = df_data[(df_data['data_source']==base_data_source)]
    df_data = df_data.drop_duplicates(subset=['data_source_index', 'timestamp'])

    df_street_segments = df_street_segments_unfiltered[(df_street_segments_unfiltered['data_source']==base_data_source)]

    df_street_segments = df_street_segments.drop_duplicates(subset=['street_segment_id'])

    ## transforming the geometries to shapely objects
    df_street_segments['street_segment_geom'] = df_street_segments['street_segment_geometry'].apply(lambda x : shapely.wkt.loads(x))
    gdf_street_segments = gpd.GeoDataFrame(df_street_segments, geometry='street_segment_geom')

    df_data['locationrange_geom'] = df_data['locationrange'].apply(lambda x : shapely.wkt.loads(x))
    gdf_data = gpd.GeoDataFrame(df_data, geometry='locationrange_geom')

    df_intersections['geom'] = df_intersections['geometry'].apply(lambda x : shapely.wkt.loads(x))
    gdf_intersections = gpd.GeoDataFrame(df_intersections, geometry='geom')

    ## Filtering for the test set itself
    ### determining the points in the prefered data_source ranges
    df_internalPoints = pd.DataFrame({'E':E,'N':N,'id':np.arange(len(N))})
    df_internalPoints['geom'] = [shapely.geometry.Point(tuple.E,tuple.N) for tuple in df_internalPoints.itertuples()]
    gdf_internalPoints = gpd.GeoDataFrame(df_internalPoints, geometry='geom')
    ### searching which source ranges contain these points
    # gdf_data = gdf_data[gdf_data['locationrange_geom'].contains(gdf_internalPoints['geom'])]
    # gdf_data = gdf_data[gdf_data['data_source_index'] in [gdf_data['locationrange_geom'].contains(tuple.geom) for tuple in gdf_internalPoints.itertuples()]]

    gdf_data['used'] = [False]*len(gdf_data.index)

    # VERY SLOW
    for tuple in gdf_internalPoints.itertuples():
        gdf_data[tuple.id] = gdf_data['locationrange_geom'].contains(tuple.geom)
    
    """# get list of data_source_index for the debugging of osqp:
    data_sources = []
    for tuple in gdf_internalPoints.itertuples():
        data_source = gdf_data[gdf_data[tuple.id]==True].iloc[0]['data_source_index']
        data_sources.append(data_source)
    print(data_sources)
    # raise SystemExit"""

    for row in gdf_data.itertuples():
        gdf_data.loc[row.Index,'used'] = any(gdf_data.loc[row.Index,np.arange(len(gdf_internalPoints.index))])

    gdf_data = gdf_data[gdf_data['used']==True]
    gdf_data.drop(columns=np.arange(len(gdf_internalPoints.index)),inplace=True)
    gdf_data.drop(columns=['used'],inplace=True)

    ### searching the streetsegments contained by the found ranges (with buffer 10^-6)
    data_source_indices = gdf_data.drop_duplicates(subset=['data_source_index'])['data_source_index'].tolist()
    # gdf_street_segments = gdf_street_segments[(gdf_street_segments.intersects(gdf_data[gdf_data['data_source_index']==data_source_indices[0]]['locationrange_geom'].iloc[0].buffer(10**(-6))))|\
    #     (gdf_street_segments.intersects(gdf_data[gdf_data['data_source_index']==data_source_indices[1]]['locationrange_geom'].iloc[0].buffer(10**(-6))))]
    list_polygon = []
    for data_source in data_source_indices:
        gdf_street_segments[data_source] = gdf_street_segments.intersects(gdf_data[gdf_data['data_source_index']==data_source]['locationrange_geom'].iloc[0].buffer(10**(-6)))
        gdf_intersections[data_source] = gdf_intersections['geom'].within(gdf_data[gdf_data['data_source_index']==data_source]['locationrange_geom'].iloc[0].buffer(10**(-6)))
        list_polygon.append(gdf_data[gdf_data['data_source_index']==data_source]['locationrange_geom'].iloc[0])

    for row in gdf_street_segments.itertuples(): # SLOW!!
        gdf_street_segments.loc[row.Index,'used'] = any(gdf_street_segments.loc[row.Index,data_source_indices])

    gdf_street_segments = gdf_street_segments[gdf_street_segments['used']==True]
    gdf_street_segments.drop(columns=data_source_indices,inplace=True)
    gdf_street_segments.drop(columns=['used'],inplace=True)

    ### searching the intersections from the found street segments
    df_segment_ids = gdf_intersections['street_segment_id'].astype(int)
    street_segment_ids = (gdf_street_segments['street_segment_id']).tolist()
    gdf_intersections = gdf_intersections[(df_segment_ids.isin(street_segment_ids))]
    # ### removing intersections outside of selected source ranges (with buffer 10^-6)
    # gdf_intersections = gdf_intersections[(gdf_intersections['geom'].within(gdf_data[gdf_data['data_source_index']==data_source_indices[0]]['locationrange_geom'].iloc[0].buffer(10**(-6))))|\
    #     (gdf_intersections['geom'].within(gdf_data[gdf_data['data_source_index']==data_source_indices[1]]['locationrange_geom'].iloc[0].buffer(10**(-6))))]
    
    for row in gdf_intersections.itertuples(): # SLOW!!
        gdf_intersections.loc[row.Index,'used'] = any(gdf_intersections.loc[row.Index,data_source_indices])

    gdf_intersections = gdf_intersections[gdf_intersections['used']==True]
    gdf_intersections.drop(columns=data_source_indices,inplace=True)
    gdf_intersections.drop(columns=['used'],inplace=True)


    df_selected_street_segments = df_street_segments_unfiltered.copy()
    df_selected_street_segments = df_selected_street_segments[(df_selected_street_segments['street_segment_id'].astype(int).isin(street_segment_ids))]
    df_selected_street_segments['street_segment_geom'] = df_selected_street_segments['street_segment_geometry'].apply(lambda x : shapely.wkt.loads(x))
    gdf_selected_street_segments = gpd.GeoDataFrame(df_selected_street_segments, geometry='street_segment_geom')
    ## determining the is_edge attribute
    # data_polygon  = shapely.ops.cascaded_union([gdf_data.loc[gdf_data['data_source_index']==data_source_indices,'locationrange_geom'].iloc[0]])
    data_polygon = shapely.ops.cascaded_union(list_polygon)
    gdf_selected_street_segments['is_edge'] = ~ gdf_selected_street_segments['street_segment_geom'].within(data_polygon.buffer(10**-6))
    gdf_intersections['is_edge'] = gdf_intersections['geom'].within(data_polygon.boundary.buffer(10**-6))
    print('DONE filtering and transforming data')


    # STEP2 visualize (when set to True)
    if visual :
        fig, ax = plt.subplots(figsize=(14,14))
        ax.set_aspect('equal')
        gdf_selected_street_segments[(~ gdf_selected_street_segments['is_edge'])].plot(ax=ax)
        gdf_selected_street_segments[(gdf_selected_street_segments['is_edge'])].plot(ax=ax, colors='k')
        # gdf_data[(gdf_data['data_source_index']==data_source_indices[0])]['locationrange_geom'].boundary.plot(ax=ax,colors='r')
        # gdf_data[(gdf_data['data_source_index']==data_source_indices[1])]['locationrange_geom'].boundary.plot(ax=ax,colors='b')
        gdf_data['locationrange_geom'].boundary.plot(ax=ax,colors='r')
        gdf_intersections[(~gdf_intersections['is_edge'])].plot(ax=ax)
        gdf_intersections[(gdf_intersections['is_edge'])].plot(ax=ax, color='k')
        # x, y = data_polygon.exterior.xy
        # plt.plot(x, y, c="green")
        # gdf_internalPoints.plot(ax=ax)
        plt.show()
        print('DONE visualising test_set')

    # STEP3 writing the data in files
    ## Only filtering on the streets without loss of other sources   
    if writing:
        df_data_source_ids = df_selected_street_segments[(~df_selected_street_segments['is_edge'])]
        data_source_ids = df_data_source_ids.drop_duplicates(subset=['data_source_index'])['data_source_index'].tolist()
        df_selected_data = df_data_unfiltered.copy()
        df_selected_data = df_selected_data[(df_selected_data['data_source_index'].astype(int).isin(data_source_ids))]
        if changed_index:
            df_selected_data.rename({'data_source_index': 'index'},axis=1,inplace=True)

        writing_path = path.join(THIS_DIR, pardir, pardir, pardir, foldername_writing)
        if path.exists(writing_path)==False:
            mkdir(path.join(THIS_DIR, pardir, pardir, pardir, foldername_writing))
            print('Had to make the directory for the results')

        df_selected_street_segments.to_csv(path.join(writing_path,'street_segments.csv'))
        gdf_intersections.to_csv(path.join(writing_path,'intersections.csv'))
        df_selected_data.to_csv(path.join(writing_path,'all_data_small_cropland_velo.csv'))
        print('DONE writing files')

if __name__ == "__main__":
    #defining the coordinates
    # E1 = 4.42
    # N1 = 51.2075
    # E2 = 4.43
    # N2 = 51.2025
    # E3 = 4.415
    # N3 = 51.1975

    # E = [E1,E2,E3,E1,E2,E3,E1,E2,E3]
    # N = [N1,N1,N1,N2,N2,N2,N3,N3,N3]

    N1 = 51.2250
    N2 = 51.2200
    N3 = 51.2150
    N4 = 51.2100
    N5 = 51.2050

    E1 = 4.4100
    E2 = 4.4150
    E3 = 4.4200
    E4 = 4.4300
    E5 = 4.4350
    E = [E1,E2,E3,E4,E5,E1,E2,E3,E4,E5,E1,E2,E3,E4,E5,E1,E2,E3,E4,E5,E1,E2,E3,E4,E5]
    N = [N1,N1,N1,N1,N1,N2,N2,N2,N2,N2,N3,N3,N3,N3,N3,N4,N4,N4,N4,N4,N5,N5,N5,N5,N5]

    create_test_set(E,N,True,True)