#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:42:57 2025

@author: au740615
"""

import pandas as pd 

infile3  ="https://pasta.lternet.edu/package/data/eml/knb-lter-ntl/129/40/af29c64a7de5ad797b5709b5f7718cb9".strip() 
infile3  = infile3.replace("https://","http://")
                 
dt3 =pd.read_csv(infile3 
          ,storage_options={'User-Agent':'EDI_CodeGen'}
          ,skiprows=1
            ,sep=","  
                ,quotechar='"' 
           , names=[
                    "year4",     
                    "sampledate",     
                    "sampletime",     
                    "air_temp",     
                    "flag_air_temp",     
                    "rel_hum",     
                    "flag_rel_hum",     
                    "wind_speed",     
                    "flag_wind_speed",     
                    "wind_dir",     
                    "flag_wind_dir",     
                    "chlor_rfu",     
                    "flag_chlor_rfu",     
                    "phyco_rfu",     
                    "flag_phyco_rfu",     
                    "do_raw",     
                    "flag_do_raw",     
                    "do_sat",     
                    "flag_do_sat",     
                    "do_wtemp",     
                    "flag_do_wtemp",     
                    "pco2_ppm",     
                    "flag_pco2_ppm",     
                    "par",     
                    "flag_par",     
                    "par_below",     
                    "flag_par_below",     
                    "ph",     
                    "flag_ph",     
                    "fdom",     
                    "flag_fdom",     
                    "turbidity",     
                    "flag_turbidity",     
                    "spec_cond",     
                    "flag_spec_cond"    ]
# data type checking is commented out because it may cause data
# loads to fail if the data contains inconsistent values. Uncomment 
# the following lines to enable data type checking
         
#            ,dtype={ 
#             'year4':'int' , 
#             'sampledate':'str' , 
#             'sampletime':'str' , 
#             'air_temp':'float' ,  
#             'flag_air_temp':'str' , 
#             'rel_hum':'float' ,  
#             'flag_rel_hum':'str' , 
#             'wind_speed':'float' ,  
#             'flag_wind_speed':'str' , 
#             'wind_dir':'float' ,  
#             'flag_wind_dir':'str' , 
#             'chlor_rfu':'float' ,  
#             'flag_chlor_rfu':'str' , 
#             'phyco_rfu':'float' ,  
#             'flag_phyco_rfu':'str' , 
#             'do_raw':'float' ,  
#             'flag_do_raw':'str' , 
#             'do_sat':'float' ,  
#             'flag_do_sat':'str' , 
#             'do_wtemp':'float' ,  
#             'flag_do_wtemp':'str' , 
#             'pco2_ppm':'float' ,  
#             'flag_pco2_ppm':'str' , 
#             'par':'float' ,  
#             'flag_par':'str' , 
#             'par_below':'float' ,  
#             'flag_par_below':'str' , 
#             'ph':'float' ,  
#             'flag_ph':'str' , 
#             'fdom':'float' ,  
#             'flag_fdom':'str' , 
#             'turbidity':'float' ,  
#             'flag_turbidity':'str' , 
#             'spec_cond':'float' ,  
#             'flag_spec_cond':'str'  
#        }
          ,parse_dates=[
                        'sampledate',
                        'sampletime',
                ] 
    )
# Coerce the data into the types specified in the metadata 
dt3.year4=pd.to_numeric(dt3.year4,errors='coerce',downcast='integer') 
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below. 
dt3=dt3.assign(sampledate_datetime=pd.to_datetime(dt3.sampledate,errors='coerce')) 
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below. 
dt3=dt3.assign(sampletime_datetime=pd.to_datetime(dt3.sampletime,errors='coerce')) 
dt3.air_temp=pd.to_numeric(dt3.air_temp,errors='coerce')  
dt3.flag_air_temp=dt3.flag_air_temp.astype('category') 
dt3.rel_hum=pd.to_numeric(dt3.rel_hum,errors='coerce')  
dt3.flag_rel_hum=dt3.flag_rel_hum.astype('category') 
dt3.wind_speed=pd.to_numeric(dt3.wind_speed,errors='coerce')  
dt3.flag_wind_speed=dt3.flag_wind_speed.astype('category') 
dt3.wind_dir=pd.to_numeric(dt3.wind_dir,errors='coerce')  
dt3.flag_wind_dir=dt3.flag_wind_dir.astype('category') 
dt3.chlor_rfu=pd.to_numeric(dt3.chlor_rfu,errors='coerce')  
dt3.flag_chlor_rfu=dt3.flag_chlor_rfu.astype('category') 
dt3.phyco_rfu=pd.to_numeric(dt3.phyco_rfu,errors='coerce')  
dt3.flag_phyco_rfu=dt3.flag_phyco_rfu.astype('category') 
dt3.do_raw=pd.to_numeric(dt3.do_raw,errors='coerce')  
dt3.flag_do_raw=dt3.flag_do_raw.astype('category') 
dt3.do_sat=pd.to_numeric(dt3.do_sat,errors='coerce')  
dt3.flag_do_sat=dt3.flag_do_sat.astype('category') 
dt3.do_wtemp=pd.to_numeric(dt3.do_wtemp,errors='coerce')  
dt3.flag_do_wtemp=dt3.flag_do_wtemp.astype('category') 
dt3.pco2_ppm=pd.to_numeric(dt3.pco2_ppm,errors='coerce')  
dt3.flag_pco2_ppm=dt3.flag_pco2_ppm.astype('category') 
dt3.par=pd.to_numeric(dt3.par,errors='coerce')  
dt3.flag_par=dt3.flag_par.astype('category') 
dt3.par_below=pd.to_numeric(dt3.par_below,errors='coerce')  
dt3.flag_par_below=dt3.flag_par_below.astype('category') 
dt3.ph=pd.to_numeric(dt3.ph,errors='coerce')  
dt3.flag_ph=dt3.flag_ph.astype('category') 
dt3.fdom=pd.to_numeric(dt3.fdom,errors='coerce')  
dt3.flag_fdom=dt3.flag_fdom.astype('category') 
dt3.turbidity=pd.to_numeric(dt3.turbidity,errors='coerce')  
dt3.flag_turbidity=dt3.flag_turbidity.astype('category') 
dt3.spec_cond=pd.to_numeric(dt3.spec_cond,errors='coerce')  
dt3.flag_spec_cond=dt3.flag_spec_cond.astype('category') 
      
print("Here is a description of the data frame dt3 and number of lines\n")
print(dt3.info())
print("--------------------\n\n")                
print("Here is a summary of numerical variables in the data frame dt3\n")
print(dt3.describe())
print("--------------------\n\n")                
                         
print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")                 

print(dt3.year4.describe())               
print("--------------------\n\n")
                    
print(dt3.sampledate.describe())               
print("--------------------\n\n")
                    
print(dt3.sampletime.describe())               
print("--------------------\n\n")
                    
print(dt3.air_temp.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_air_temp.describe())               
print("--------------------\n\n")
                    
print(dt3.rel_hum.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_rel_hum.describe())               
print("--------------------\n\n")
                    
print(dt3.wind_speed.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_wind_speed.describe())               
print("--------------------\n\n")
                    
print(dt3.wind_dir.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_wind_dir.describe())               
print("--------------------\n\n")
                    
print(dt3.chlor_rfu.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_chlor_rfu.describe())               
print("--------------------\n\n")
                    
print(dt3.phyco_rfu.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_phyco_rfu.describe())               
print("--------------------\n\n")
                    
print(dt3.do_raw.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_do_raw.describe())               
print("--------------------\n\n")
                    
print(dt3.do_sat.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_do_sat.describe())               
print("--------------------\n\n")
                    
print(dt3.do_wtemp.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_do_wtemp.describe())               
print("--------------------\n\n")
                    
print(dt3.pco2_ppm.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_pco2_ppm.describe())               
print("--------------------\n\n")
                    
print(dt3.par.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_par.describe())               
print("--------------------\n\n")
                    
print(dt3.par_below.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_par_below.describe())               
print("--------------------\n\n")
                    
print(dt3.ph.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_ph.describe())               
print("--------------------\n\n")
                    
print(dt3.fdom.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_fdom.describe())               
print("--------------------\n\n")
                    
print(dt3.turbidity.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_turbidity.describe())               
print("--------------------\n\n")
                    
print(dt3.spec_cond.describe())               
print("--------------------\n\n")
                    
print(dt3.flag_spec_cond.describe())               
print("--------------------\n\n")

import datetime as dt 
filtered_df = dt3[dt3['year4'] == 2022]    
filtered_df['sampletime_datetime'] = pd.to_datetime(dt3['sampletime_datetime']).dt.time

filtered_df = filtered_df[["sampledate_datetime", "sampletime_datetime", "chlor_rfu"]]
filtered_df.to_csv('/Users/au740615/Documents/projects/ecokgml-example/processed_data.csv', index=False)