# =============================================
#
# TERMS AND CONDITIONS
# FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# Apache License 2.0
#
# =============================================

# *********************************************
#
# Author:
#   2020 MCM/ICM Team 2014906
#
# Last Update:
#   Feb. 17, 2020
#
# *********************************************


import datetime
import netCDF4 as nc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import folium as fo
from scipy.interpolate import interp1d

# ==========
# Handle Raw Data
# ==========
file = r"data/20060116_21001216_month.nc"
# file = r"data/18500702_28490702_year.nc"
ds = nc.Dataset(file)

lat_range = (42, 69)  # 42.3N - 68.8N (-78,90)
lon_range = (316, 360)  # 43.8W - 0 (1,360)
time_range = (datetime.datetime.strptime("2006-01-01", "%Y-%m-%d"),
              datetime.datetime.strptime("2020-01-01", "%Y-%m-%d"))  # [15.5, 45.0, 74.5,..., 34629.0, 34659.5]
base_time = datetime.datetime.strptime("2006-01-01", "%Y-%m-%d")
days_range = ((time_range[0] - base_time).days, (time_range[1] - base_time).days)

# --------------------------------------------------
# Some codes to plot the global SST of July, 2006
# --------------------------------------------------

# Filter out SST of Target Latitude, Lontitude, Time Range
lat_range_idx = np.ma.where((lat_range[0] <= ds["lat"][:]) * (lat_range[1] >= ds["lat"][:]))
lon_range_idx = np.ma.where((ds["lon"][:] >= lon_range[0]) * (ds["lon"][:] <= lon_range[1]))
days_range_idx = np.ma.where((ds["time"][:] >= days_range[0]) * (ds["time"][:] <= days_range[1]))
_lat = ds["lat"][:][lat_range_idx]
_lon = ds["lon"][:][lon_range_idx]
z = ds["tos"][:][days_range_idx][6::12][:, min(lat_range_idx[0]): max(lat_range_idx[0]) + 1, :][...,
    min(lon_range_idx[0]):max(lon_range_idx[0]) + 1]  # Only July!

print("==========\nHandling Raw Data Results"
      "\n\t%d\tmonth-intervals\n\t%d\tlatitude-intervals\n\t%d\tlontitude-intervals"
      "\n==========" % (z.shape))

# ==========
# Gray Prediction - 2006.7 to 2070.7， 64
# ==========
PREDICTION_MONTH_CNT = 64
print("\n==========\nGrey Prediction")
# +++ I. Find & Check the Step Ratio lambda(k)
lambds = z[:-1, ...] / z[1:, ...]  # Actually is lambda.T
valid_range = (np.exp(-2.0 / (z.shape[0] + 1)), np.exp(2.0 / (z.shape[0] + 2)))
validation = ((lambds > valid_range[0]) * (lambds < valid_range[1])).all()
if validation:
    print("Initially Valid")
else:
    print("Initially Invalid")
    # Move by c = delta
    done = False
    test_lim = 1e7
    for delta in np.arange(-test_lim, test_lim, max(1, int(test_lim / 2000.))):
        lambds = (z[:-1, ...] + delta) / (z[1:, ...] + delta)
        validation = ((lambds > valid_range[0]) * (lambds > valid_range[1])).all()
        if validation:
            print("Valid after Moving Date Points by %.3g" % delta)
            done = True
            break
    if not done:
        print("Still Invalid")
        exit(0)

# +++ II. GM(1,1) Modeling - Point-Oriented
# ------ II.1. Sum Up
# ------ II.2. B, Y Formation
# ------ II.3. Relative Error
predict_ori = np.zeros(z.shape)
predict_fifty = np.zeros((PREDICTION_MONTH_CNT, z.shape[1], z.shape[2]))
for lat_idx in range(z.shape[1]):
    for lon_idx in range(z.shape[2]):
        # sum up
        this_trend = np.ma.getdata(z[:, lat_idx, lon_idx])
        this_trend_sum = np.cumsum(this_trend, axis=0)

        # B, Y formation
        B = np.ones((this_trend_sum.shape[0] - 1) * 2).reshape((this_trend_sum.shape[0] - 1, 2))
        B[:, 0] = (this_trend_sum[:-1] + this_trend_sum[1:]) / -2.0
        Y = this_trend[1:].reshape(-1, 1)
        # calculate u_vector (pay attention to the types: masked or not)
        u_vector = np.linalg.inv(B.T @ B) @ (B.T) @ Y

        # calculate prediction: shape 1*PREDICTION_MONTH_CNT = 1*768
        model_sol = np.zeros(PREDICTION_MONTH_CNT)
        b_div_a = u_vector[1] / float(u_vector[0])
        model_sol[1:] = (this_trend[0] - b_div_a) * \
                        np.exp(-u_vector[0] * np.arange(1, PREDICTION_MONTH_CNT, 1)) \
                        + b_div_a
        model_sol[0] = this_trend[0]
        this_predict = np.zeros(PREDICTION_MONTH_CNT)
        this_predict[1:] = model_sol[1:] - model_sol[:-1]
        this_predict[0] = this_trend[0]

        # store result
        predict_fifty[:, lat_idx, lon_idx] = this_predict

predict_ori = predict_fifty[:z.shape[0], ...]
print("Prediction Shape:", predict_ori.shape, predict_fifty.shape)

# Calculate & Plot the relative difference 2006.1.1 to 2020.1.1 (%): (origin-prediction)/origin*100%
# [Calculate]
relative_dif = (z - predict_ori) / z * 100
plt_diff_1 = np.max(relative_dif, axis=0)
plt_diff_2 = np.min(relative_dif, axis=0)
plt_dif_thres = 10
plt_change_idx = np.where((plt_diff_1 > plt_dif_thres) * (plt_diff_1 < -plt_dif_thres) *
                          (plt_diff_2 > plt_dif_thres) * (plt_diff_2 < -plt_dif_thres))
plt_diff_1[plt_change_idx] = 0
plt_diff_2[plt_change_idx] = 0
plt_x, plt_y = np.meshgrid(_lon, _lat)
plt_diff = [plt_diff_1, plt_diff_2]

# [Plot]
cmap = [cm.OrRd, cm.winter]  # [cm.terrain, cm.ocean]
titles = ["Upper Bounds", "Lower Bounds"]
limits = [(0, 1.4), (-1, -0)]
for i in range(2):
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(plt_x, plt_y, plt_diff[i], cmap=cmap[i], linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.8)  # , aspect=20)
    ax.set_title(titles[i])
    ax.set_xlabel(r"Longitude / $\circ$W"), ax.set_ylabel(r"Latitude / $\circ$N"), ax.set_zlabel(r"Error / %")
    plt_xticks = np.arange(320, 370, 10)
    ax.set_xticks(plt_xticks), ax.set_xticklabels(360 - plt_xticks)
    ax.set_zlim(limits[i])
    fig.tight_layout()
    plt.show()
print("Cal & Plot Relative Error")
print("==========")

# ==========
# Global Warming
# ==========
print("\n==========\nGlobal Warming")
k = 0.033091716476687454
b = -0.05923436411387261
warming_delta_temp = k * np.arange(0, PREDICTION_MONTH_CNT, 1) + b
for _pre_yr in range(PREDICTION_MONTH_CNT):
    predict_fifty[_pre_yr, ...] += warming_delta_temp[_pre_yr]
print("==========")

# --------------------------------------------------
plt.clf()
fig = plt.figure(figsize=(6, 4))
to_plot = predict_fifty[-1].copy()
cf = plt.contourf(_lon, _lat, np.ma.masked_array(to_plot, mask=np.where(to_plot > 200, False, True)),
                  cmap=plt.cm.coolwarm)
fig.colorbar(cf, label="SST / K")
plt.xticks(np.arange(320, 365, 5), 360 - np.arange(320, 365, 5))
plt.ylabel("Latitude / °N"), plt.xlabel("Longitude / °W")
fig.subplots_adjust(left=0.10, right=1.0, top=0.90, bottom=0.12)
plt.title("Predicted SST in July, 2070")
plt.show()
# --------------------------------------------------

# --------------------------------------------------
# Some codes to plot SST Difference in July, Year 2020 and 2070")
# --------------------------------------

# ==========
# Movement - Circle Center
# ==========
print("\n==========\nMovement")
harbours = {"name": ["Orkney", "Scrabster", "Kinlochbervie", "Stornoway", "Lochinver",
                     "Ullapool", "Portree", "Buckie", "Fraserburgh", "Peterhead"],
            "lat": np.array(
                [58.92421, 58.610807, 58.458548, 58.210575, 58.147846,
                 57.895426, 57.410842, 57.680039, 57.692437, 57.502806]),
            "lon": np.array([-2.747102, -3.552002, -5.050404, -6.384936, -5.243184,
                             -5.15997, -6.190662, -2.956613, -2.003744, -1.774236])}
trace = {"name": harbours['name'],
         "lat": [], "lat_idx": [],
         "lon": [], "lon_idx": [],
         "year": [], "looped_yr": []}
year_base = 13  # constant
threshold = 0.05  # constant
mask = np.ma.getmaskarray(ds["tos"][:])

for harb_idx in range(len(harbours["name"])):
    loc = (round(harbours["lat"][harb_idx]), round(harbours["lon"][harb_idx] + 360))
    loc_idx = np.array([int(np.where(_lat == loc[0])[0]), int(np.where(_lon == loc[1])[0])])  # init

    year_cnt = 0  # init
    temperature = predict_fifty[year_base, loc_idx[0], loc_idx[1]]  # init
    locations = [np.array([harbours["lat"][harb_idx], harbours["lon"][harb_idx] + 360])]  # init as results, by lat*lon
    locations_idx = [loc_idx]  # init as results, by idx
    locations_change_yr = [2019]
    while year_cnt < 50:
        if (loc_idx[0] + 1 > len(_lat) - 1 or loc_idx[0] - 1 < 0) or (
                loc_idx[1] + 1 > len(_lon) - 1 or loc_idx[1] - 1 < 0):
            break
        new_loc_idx = [(loc_idx[0] - 1, loc_idx[1] + 1), (loc_idx[0], loc_idx[1] + 1),
                       (loc_idx[0] + 1, loc_idx[1] + 1),
                       (loc_idx[0] - 1, loc_idx[1]), (loc_idx[0] + 1, loc_idx[1]),
                       (loc_idx[0] - 1, loc_idx[1] - 1), (loc_idx[0], loc_idx[1] - 1),
                       (loc_idx[0] + 1, loc_idx[1] + 1)]
        delta = []
        for i in new_loc_idx:
            # if mask[year_cnt + year_base + 1, i[0], i[1]]:
            delta.append([predict_fifty[year_cnt + year_base + 1, i[0], i[1]] - temperature])
            # else:
            #     delta.append(1e20)
        abs_delta = np.abs(delta)
        poss_next_loc = np.argmin(abs_delta)
        if abs_delta[poss_next_loc] < threshold:
            loc_idx = np.array(new_loc_idx[poss_next_loc])
            loc = np.array([_lat[loc_idx[0]], _lon[loc_idx[1]]])
            temperature = predict_fifty[year_cnt + year_base + 1, loc_idx[0], loc_idx[1]]
            locations_idx.append(loc_idx)

            # prevent (x,y) points with the same x value
            for loc_delta in np.arange(0, 1, 0.1):
                temp_loc = loc + loc_delta
                loc_nondup = True
                for s_loc in locations:
                    if (s_loc == temp_loc).all():
                        loc_nondup = False
                        break
                if loc_nondup:
                    locations.append(temp_loc)
                    break

            locations_change_yr.append(year_cnt + year_base + 1 + 2006)
        year_cnt += 1

    locations = np.array(locations)
    locations_idx = np.array(locations_idx)
    locations_change_yr = np.array(locations_change_yr)

    trace["lat"].append(locations[:, 0])
    trace["lon"].append(locations[:, 1] - 360)
    trace["lat_idx"].append(locations_idx[:, 0])
    trace["lon_idx"].append(locations[:, 1])
    trace["year"].append(locations_change_yr)
    trace["looped_yr"].append(year_cnt)
print("==========")

# ==========
# Movement - Draw Map
# ==========
print("\n==========\nMap")
colors = ["green", "red", "blue", "gray"]
target_harb_idx = [0, 1, 7, 8, 9]
for harb_idx in target_harb_idx:
    map = fo.Map(location=[60.4, -3.1],
                 zoom_control=False, zoom_start=6)

    zipped_pos = zip(trace["lat"][harb_idx], trace["lon"][harb_idx])
    fo.PolyLine(locations=np.array(list(zipped_pos))).add_to(map)

    for pos_i in range(len(trace["lat"][harb_idx])):
        pos_lat = trace["lat"][harb_idx][pos_i]
        pos_lon = trace["lon"][harb_idx][pos_i]
        pos_year = str(trace["year"][harb_idx][pos_i])
        if 0 == pos_i:
            fo.Marker(location=(pos_lat, pos_lon), popup=fo.Popup(pos_year),
                      icon=fo.Icon(color="cadetblue", icon="ship", prefix="fa")).add_to(map)
        elif len(trace["lat"][harb_idx]) - 1 == pos_i:
            fo.Marker(location=(pos_lat, pos_lon), popup=fo.Popup(pos_year),
                      icon=fo.Icon(color="lightred", icon="cloud")).add_to(map)
        else:
            fo.Marker(location=(pos_lat, pos_lon), popup=fo.Popup(pos_year)).add_to(map)
    fo.Circle(location=(trace["lat"][harb_idx][-1], trace["lon"][harb_idx][-1]), radius=5e3)

    map.save("plots/" + trace["name"][harb_idx] + ".html")
print("==========")


# ==========
# Calculate the shortest distance between two given location
# ==========
def cal_dist(lat1, lon1, lat2, lon2):
    R = 6371  # kilometers
    theta1, theta2, delta_theta, delta_lamb = np.radians([lat1, lat2, lat1 - lat2, lon1 - lon2])
    a = np.sin(delta_theta / 2) * np.sin(delta_theta / 2) + \
        np.cos(theta1) * np.cos(theta2) * np.sin(delta_lamb / 2) * np.sin(delta_lamb / 2)
    dist = 2 * R * np.arcsin(np.sqrt(a))
    return dist


# ==========
# Calculate & Plot the yearly distance between fish schools and ports
# ==========
print("\n==========\nDistance")
plt.clf()
fig = plt.figure(figsize=(8, 4))
# target_harb_idx = [0, 1, 6, 7, 8, 9]
colors = plt.get_cmap('nipy_spectral')(np.linspace(0.1, 0.9, len(target_harb_idx)))
range_threshold = [250, 450]
range_thres_year_first = {"name": [], "250": [], "450": []}
range_thres_year_last = {"name": [], "250": [], "450": []}
for target_idx, harb_idx in enumerate(target_harb_idx):
    harb_name = trace["name"][harb_idx]
    harb_yr = trace["year"][harb_idx]
    harb_loc = (trace["lat"][harb_idx][0], trace["lon"][harb_idx][0])
    harb_dist = []
    fish_lats, fish_lons = list(trace["lat"][harb_idx]), list(trace["lon"][harb_idx])
    harb_yr = list(harb_yr)
    if harb_yr[-1] < 2070:
        fish_lats.append(fish_lats[-1])
        fish_lons.append(fish_lons[-1])
        harb_yr.append(2070)

    # Cubic Interpolation
    plt_yr = np.linspace(2019, 2070, endpoint=True)
    f_intpl_lat = interp1d(harb_yr, fish_lats, kind='cubic')
    f_intpl_lon = interp1d(harb_yr, fish_lons, kind='cubic')
    intpl_lat = f_intpl_lat(plt_yr)
    intpl_lon = f_intpl_lon(plt_yr)

    # calculate the distances, w.r.t interpolated latitudes & longitudes
    for yr_lat, yr_lon in zip(intpl_lat, intpl_lon):
        harb_dist.append(cal_dist(harb_loc[0], harb_loc[1], yr_lat, yr_lon))
    plt.plot(plt_yr, harb_dist, label=harb_name, color=colors[target_idx])

    # calculate boundary years
    range_thres_year_first["name"].append(harb_name)
    range_thres_year_last["name"].append(harb_name)
    for range_bnd in range_threshold:
        range_thres_year_first[str(range_bnd)].append(
            int(round(plt_yr[np.min(np.where(np.array(harb_dist) >= range_bnd))])))
        range_thres_year_last[str(range_bnd)].append(
            int(round(plt_yr[np.max(np.where(np.array(harb_dist) <= range_bnd))])))

plt.xticks(np.arange(2020, 2075, 5))
plt.xlabel("Year"), plt.ylabel("Fish School Distance / km")
plt.title("Predicted Fish School Distance to 5 Ports in the Next 50 Years")
plt.grid(linestyle="dashed")
plt.legend(loc="lower right")  # plt.legend(loc="best")
fig.subplots_adjust(left=0.10, right=0.95, top=0.90)
plt.show()

print("==========")
