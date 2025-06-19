# Use an official Node.js runtime as the base image
FROM node:20

# Set the working directory in the Docker image
WORKDIR /usr/src/app

# Copy package.json and package-lock.json into the Docker image
COPY package*.json ./

# Install the application's dependencies inside the Docker image
RUN npm install

# Copy the rest of the application code into the Docker image
COPY . .

# Expose port 6685 to the outside world
EXPOSE 6685

# Start the application
CMD [ "node", "encryptMB.js" ]
