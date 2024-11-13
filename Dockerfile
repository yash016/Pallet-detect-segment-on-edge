# Use the official ROS 2 base image
FROM osrf/ros:jazzy-desktop-full

# Set the working directory
WORKDIR /ros_ws

# Copy the ROS package into the workspace
COPY src/ ./src/

# Copy and install Python dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Source ROS 2 environment and build the workspace
RUN . /opt/ros/jazzy/setup.sh && colcon build

# Source the setup script
RUN echo "source /ros_ws/install/setup.bash" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

# Set the entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
