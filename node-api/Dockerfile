FROM node:22-alpine
WORKDIR /app
COPY node-api/package.json /app/package.json
RUN npm install
COPY node-api/src /app/src
EXPOSE 4000
CMD ["npm", "start"]
