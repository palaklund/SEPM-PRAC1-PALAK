# Base image
FROM node:18

# Set working directory
WORKDIR /app

# Copy files
COPY package.json .
COPY app.js .

# Install dependencies (none here, but good practice)
RUN npm install

# Expose port
EXPOSE 5005

# Run app
CMD ["node", "app.js"]