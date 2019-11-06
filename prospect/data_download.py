import requests
import io, os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time

from multiprocessing.pool import ThreadPool
import pandas as pd

################################################################################
df = pd.read_csv('./data/source_minedex/Sites.csv', encoding = "ISO-8859-1")

names = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Commodities']
data = df[names].dropna()
data_arr = data.values

################################################################################
# save data for future use
# input_df = pd.DataFrame( np.delete(data_arr, [17207, 39289, 39290 ],0), columns = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Commodities'])
# input_df.to_csv('num_data.csv')

# output = data['Commodities'].str.contains('Au', case=False)
# # output = pd.Series([name == 'Au' for name in data['Commodities'].values])
# output_arr = np.delete(np.array(output.values, dtype=int), [17207, 39289, 39290 ])
# output_df = pd.DataFrame(output_arr, columns=['gold'])
# output_df.to_csv('./data/output.csv')

################################################################################
def download_extract_zip(url):
    """
    Download a ZIP file and extract its contents in memory
    yields (filename, file-like object) pairs
    """
    try:
        response = requests.get(url[1])
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for entry in archive.infolist():
                with archive.open(entry) as file:
                    im = Image.open(file).resize((32,32), Image.ANTIALIAS)
                    # return np.array(im)
                    im.save('/Users/romka/Desktop/mine_detect/img/' + str(url[0]).zfill(6) + '.tiff')
    except:
        pass



def make_url(ind, lat, long):
    layers = ['AlOH_group_composition',
              'Ferrous_Iron_Index',
              'Opaque_Index',
              'Ferric_oxide_content',
              'FeOH_group_content',
              'Ferric_Oxide_Composition',
              'Kaolin_Group_Index',
              'Quartz_Index',
              'MgOH_Group_Content',
              'Green_Vegetation',
              'Ferrous_Iron_Content_in_MgOH',
              'MgOH_Group_Composition',
              'AlOH_Group_Content',
              'Gypsum_Index',
              'Silica_Index'
              ]

    serviceUrl = [
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_AlOH_group_composition_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Ferrous_Iron_Index_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Opaque_Index_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Ferric_oxide_content_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_FeOH_group_content_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Ferric_oxide_composition_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Kaolin_group_index_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/thermal/Aus_ASTER_L2EM_Quartz_Index_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_MgOH_group_content_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Green_vegetation_content_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_Ferrous_iron_content_in_MgOH_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_MgOH_group_composition_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/vnir/Aus_Mainland/Aus_Mainland_AlOH_group_content_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/thermal/Aus_ASTER_L2EM_Gypsum_Index_reprojected.nc4',
        'http://dap-wms.nci.org.au/thredds/wcs/wx7/aster/thermal/Aus_ASTER_L2EM_Silica_Index_reprojected.nc4'
        ]

    ftpURL = [
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/AlOH_Group_Composition/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Ferrous_Iron_Index/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Opaque_Index/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Ferric_Oxide_Content/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/FeOH_Group_Content/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Ferric_Oxide_Composition/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Kaolin_Group_Index/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_Thermal_Products/Quartz_Index/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/MgOH_Group_Content/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Green_Vegetation/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/Ferrous_Iron_Content_in_MgOH/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/MgOH_Group_Composition/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_VNIR_SWIR_Products/AlOH_Group_Content/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_Thermal_Products/Gypsum_Index/',
        'ftp://ftp.csiro.au/arrc/Australian_ASTER_Geoscience_Map/ASTER_Thermal_Products/Silica_Index/'
        ]

    url = 'http://portal.auscope.org/api/downloadWCSAsZip.do' + \
    '?layerName=' + layers[ind] + \
    '&serviceUrl=' + serviceUrl[ind] + \
    '&usingBboxConstraint=on' + \
    '&northBoundLatitude={}'.format(lat + 0.1) + \
    '&southBoundLatitude={}'.format(lat - 0.1) + \
    '&eastBoundLongitude={}'.format(long + 0.1) + \
    '&westBoundLongitude={}'.format(long - 0.1) + \
    '&outputDimensionsType=widthHeight&outputWidth=256&outputHeight=256&inputCrs=OGC:CRS84&downloadFormat=GeoTIFF' \
    '&outputCrs=EPSG:0%20%5BLatitude_Longitude%5D' + \
    '&ftpURL=' + ftpURL[ind]
    return url


# for i in range(len(layers)):
#     try:
#         url = make_url(i, -21.730934, 122.214974 , layers, service_urls, ftp_urls)
#         im = download_extract_zip(url)
#         plt.figure(i)
#         plt.imshow(im)
#         plt.show()
#         print(i, im.shape)
#     except:
#         pass



url_list = list(zip(np.arange(len(data)),[make_url(1, data['Latitude'].values[x], data['Longitude'].values[x]) for x in range(len(data))]))

temp = os.listdir('./img')
temp.sort()
new = [int(item.lstrip('0').rstrip('.tiff')) for item in temp[2:]]
total = np.arange(len(data))
cr = np.r_[[i not in new for i in total]]

index_list = total[cr]



url_list = [url_list[i] for i in index_list]


with ThreadPool(16) as p:
    p.map(download_extract_zip, url_list)

    print('done')



