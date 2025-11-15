# Calibration Monitoring

Create an hourly job to monitor the calibration of onvifcams across various kubectx clusters (as specified in devices.yaml)

1. initial 1 time - collect reference scan from camera & store in S3 - like in scan.py
2. ongoing - 
    1. extract stable frame or command a specific p/t/z after waiting 30 seconds for no camera movement (over port forward). See how MQTT telemetry is consumed in scan.py.
    2. run the algo, output pitch/yaw/tilt offset
    3. write results to athena table and slack bot

Create iceberg table `calibration_monitoring` in Athena+Glue, with columns:

- deployment_name
- device_id
- timestamp
- pitch_offset
- yaw_offset
- roll_offset
- mode: passive or active
- capture_positions (list of {pan: x, tilt: y, zoom: z})
- files_location
- success: True/False
- failure_log: any log explaining the failure (i.e. timeout passed)

Reference Collection:

1. Run camera scan, saving images in `s3://camera-calibration-monitoring/{deployment-name}/{camera-name}/reference_scan/images` 
    1. copy the relevant code from scan.py
2. Extract Features and save in `s3://camera-calibration-monitoring/{deployment-name}/{camera-name}/reference_scan/features` 
3. Repeat for each kube cluster & camera in @devices.yaml - you will have to change the kubectx and port forward

Query extraction:

1. Args: 
    1. device_id
    2. stabilize_time=30 seconds
    3. timeout=None (seconds)
    4. active_ptz_stops:List[Tuple of length 3]=[] - if not provided, will run in passive mode.
2. Logic:
    1. port forward to MQTT for device
    2. wait for camera pitch/yaw/roll to be stable (0 movement) for 30 seconds
        1. if zero movement:
            1. connect to rtsp for the camera
            2. if no ptz_stops specified, capture a frame and its telemetry
            3. if ptz_stops is specified, then command camera to go to each ptz stop and capture at each stop, precisely as done in [scan.py](http://scan.py)
        2. if timeout is specified and it elapses with no periods of 0 camera movement, then exit and log to DB and slack
    3. save frames and manifest mapping the frame to its telemetry to local file system (temporarily)
3. Return: {’camera_frames’: List[camera frames], telemetry_
    1. wait for camera to be stable for 30 seconds with no pitch/yaw/roll movement. if timeout specified, stop monitoring MQTT and end job after timeout.

Calibration Monitoring Algorithm:

1. Run the algorithm over the locally stored query files vs the reference features
2. Save the results to the Athena table
3. move the temporarily locally stored frames & telemetry manifest to `s3://camera-calibration-monitoring/{deployment-name}/{camera-name}/query_scan/{timestamp}` 
4. Send results in slack channel `calibration_monitoring`, with a threshold of 0.5 offset in either yaw/pitch/roll for alerting with a 🚨 emoji, otherwise should be a ✅ emoji