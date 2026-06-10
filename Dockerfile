# Start with an official ROS 2 base image for the desired distribution
FROM ros:humble-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble \
    RCUTILS_COLORIZED_OUTPUT=1

ARG USER_UID=1001
ARG USER_GID=1001
ARG USERNAME=user

# Install essential packages and ROS development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash-completion \
        curl \
        gdb \
        git \
        nano \
        iputils-ping \
        openssh-client \
        python3-colcon-argcomplete \
        python3-colcon-common-extensions \
        sudo \
        vim \
        libgtest-dev \
        libgmock-dev \
        libpciaccess-dev \
        iproute2 \
        libomp-dev \
        lsb-release \
        gnupg2 \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Robotpkg repository + Pinocchio with CasADi support ──────────────────────
RUN mkdir -p /etc/apt/keyrings \
    && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
       | tee /etc/apt/keyrings/robotpkg.asc \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] \
       http://robotpkg.openrobots.org/packages/debian/pub \
       $(lsb_release -cs) robotpkg" \
       | tee /etc/apt/sources.list.d/robotpkg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        robotpkg-py310-pinocchio \
        robotpkg-py310-casadi \
        robotpkg-coal \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Robotpkg environment variables (robotpkg MUST come first) ─────────────────
ENV PATH="/opt/openrobots/bin:$PATH"
ENV PKG_CONFIG_PATH="/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH"
ENV LD_LIBRARY_PATH="/opt/openrobots/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH"
ENV CMAKE_PREFIX_PATH="/opt/openrobots:$CMAKE_PREFIX_PATH"

# Setup user configuration
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/$USERNAME/.bashrc

USER $USERNAME

# Install ROS 2 dependencies
# NOTE: ros-humble-pinocchio intentionally excluded — using robotpkg version
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        ros-humble-ros-gz \
        ros-humble-sdformat-urdf \
        ros-humble-joint-state-publisher-gui \
        ros-humble-ros2controlcli \
        ros-humble-controller-interface \
        ros-humble-hardware-interface-testing \
        ros-humble-ament-cmake-clang-format \
        ros-humble-ament-cmake-clang-tidy \
        ros-humble-controller-manager \
        ros-humble-ros2-control-test-assets \
        libignition-gazebo6-dev \
        libignition-plugin-dev \
        ros-humble-hardware-interface \
        ros-humble-control-msgs \
        ros-humble-backward-ros \
        ros-humble-generate-parameter-library \
        ros-humble-realtime-tools \
        ros-humble-joint-state-publisher \
        ros-humble-joint-state-broadcaster \
        ros-humble-moveit-ros-move-group \
        ros-humble-moveit-kinematics \
        ros-humble-moveit-planners-ompl \
        ros-humble-moveit-ros-visualization \
        ros-humble-joint-trajectory-controller \
        ros-humble-moveit-simple-controller-manager \
        ros-humble-rviz2 \
        ros-humble-xacro \
        ros-humble-teleop-twist-keyboard \
        ros-humble-joy \
        ros-humble-teleop-twist-joy \
        ros-humble-velocity-controllers \
        ros-humble-ros2-controllers \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# Install workspace dependencies, then purge ROS pinocchio/hpp-fcl if pulled in
COPY . /ros2_ws/src
RUN sudo chown -R $USERNAME:$USERNAME /ros2_ws \
    && sudo apt-get update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y \
    && sudo apt-get remove -y --purge ros-humble-pinocchio ros-humble-hpp-fcl || true \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* \
    && rm -rf /home/$USERNAME/.ros

# ── Python dependencies ───────────────────────────────────────────────────────
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-visp \
        ros-humble-cv-bridge \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*
    
RUN sudo apt-get remove -y python3-sympy \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

RUN sudo pip3 install --upgrade pip \
    && sudo pip3 install --no-cache-dir --break-system-packages --timeout=300 \
        "numpy<2" \
        pillow \
        pandas \
        scipy
        
        
# Verify pinocchio.casadi is available with the correct library
RUN python3 -c "import pinocchio as pin; print('pinocchio from:', pin.__file__)" \
    && python3 -c "from pinocchio import casadi as cpin; print('pinocchio.casadi OK:', cpin.__file__)"

SHELL ["/bin/bash", "-c"]
WORKDIR /ros2_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build \
        --symlink-install \
        --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCOAL_DISABLE_HPP_FCL_WARNINGS=ON \
    && rm -rf build/libfranka/CMakeFiles \
               build/haption_raptor_api/CMakeFiles \
               build/haptic_control/CMakeFiles \
               build/test_admittance/CMakeFiles \
               build/test_calibration/CMakeFiles \
               build/test_impedance/CMakeFiles \
               build/franka_msgs/CMakeFiles \
               build/franka_description/CMakeFiles \
               build/franka_gazebo_bringup/CMakeFiles \
               build/franka_mobile_sensors/CMakeFiles \
               build/raptor_api_interfaces/CMakeFiles

RUN sudo mkdir -p /etc/Haption
COPY ./6DOFs_haptic_devices/n71/7.2.0000/SvcHaptic_Desktop_6D_n71.conf /etc/Haption/
COPY ./6DOFs_haptic_devices/n70/7.2.0000/SvcHaptic_Desktop_6D_n70.conf /etc/Haption/

COPY ./6DOFs_haptic_devices/n71/7.2.0000/Desktop_6D_n71.hap /etc/Haption/
COPY ./6DOFs_haptic_devices/n70/7.2.0000/Desktop_6D_n70.hap /etc/Haption/

RUN sudo mkdir -p /etc/Haption/Connector
COPY ./6DOFs_haptic_devices/n71/desktop_6D_n71.param /etc/Haption/Connector/
COPY ./6DOFs_haptic_devices/n70/desktop_6D_n70.param /etc/Haption/Connector/

RUN sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Bashrc: robotpkg paths re-asserted AFTER ros source to win over ROS overrides
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "# robotpkg must come first — overrides ROS pinocchio/coal" >> ~/.bashrc && \
    echo "export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:\$PYTHONPATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=20" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/ros2_ws/src/haptic_interface_ROS2/src/haption_raptor_api/Dependencies/RaptorAPI/bin/Linux/glibc-2.35/" >> ~/.bashrc && \
    echo "alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc && \
    echo -e 'shopt -s histappend\nPROMPT_COMMAND="history -a; $PROMPT_COMMAND"\nexport HISTSIZE=10000\nexport HISTFILESIZE=20000' >> ~/.bashrc

COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
