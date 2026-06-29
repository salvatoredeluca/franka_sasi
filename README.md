# Istruzioni Esecuzione Esperimento

## 1. Avvio Ambiente Docker
```bash
docker compose up -d
```

## 2. Accesso al Container
*Attenzione: per ogni step dal 3 al 7 dovrai aprire un nuovo terminale e lanciare questo comando per entrare nel container.*
```bash
docker exec -it franka_ros2_humble bash
```

## 3. Bringup Franka
```bash
ros2 launch franka_bringup franka_bringup.launch.py 
```

## 4. Nodo Controllo
```bash
ros2 run my_control_package OcclusionTask 
```

## 5. Nodo Detection
```bash
ros2 run franka_detection EdgesDetection 
```

## 6. Nodo Joypad
```bash
ros2 run joy joy_node 
```

## 7. Registrazione ROS 2 Bag
```bash
ros2 bag record -o bag_esperimento_1 /error_image_space /left/camera_left/color/image_raw /H
```
