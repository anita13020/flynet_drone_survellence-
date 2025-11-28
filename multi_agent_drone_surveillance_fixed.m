% multi_agent_drone_surveillance_fixed.m
% Multi-Agent Quantized Drone Surveillance with Coordinated Edge AI
fprintf('=== Multi-Agent Quantized Drone Surveillance ===\n');
fprintf('Starting coordinated simulation...\n\n');
% Initialize multi-agent system
multiAgentSystem = initializeMultiAgentSystem(3); % 3 drones
% Run coordinated surveillance mission
missionDuration = 20; % Reduced for faster testing
multiAgentSystem = runCoordinatedMission(multiAgentSystem, missionDuration);
% Generate multi-agent performance report
generateMultiAgentReport(multiAgentSystem);
% Multi-Agent System Initialization
function multiAgentSystem = initializeMultiAgentSystem(numDrones)
fprintf('Initializing Multi-Agent System with %d drones...\n', numDrones);
% Create empty multiAgentSystem structure
multiAgentSystem = struct();
multiAgentSystem.numDrones = numDrones;
multiAgentSystem.communicationRange = 100;
multiAgentSystem.coordinationMode = 'distributed';
multiAgentSystem.sharedMap = struct();
multiAgentSystem.missionObjectives = {};
% Create template drone structure with all required fields
templateDrone = createTemplateDrone();
% Pre-allocate drones array
multiAgentSystem.drones = repmat(templateDrone, 1, numDrones);
% Initialize individual drones
for i = 1:numDrones
multiAgentSystem.drones(i).id = i;
multiAgentSystem.drones(i).position = [randi([0, 500]), randi([0,
500]), 50 + randi([0, 50])];
multiAgentSystem.drones(i).batteryLevel = 100;
multiAgentSystem.drones(i).status = 'active';
multiAgentSystem.drones(i).role = assignDroneRole(i, numDrones);
multiAgentSystem.drones(i).capabilities = defineDroneCapabilities(i);
multiAgentSystem.drones(i).aiSystem =
initializeQuantizedDroneAI(multiAgentSystem.drones(i).role);
multiAgentSystem.drones(i).communicationBuffer = {};
multiAgentSystem.drones(i).taskQueue = {};
multiAgentSystem.drones(i).neighbors = [];
multiAgentSystem.drones(i).currentAction = 'patrol';
multiAgentSystem.drones(i).decisionConfidence = 0;
end

1

% Initialize shared situational awareness
multiAgentSystem.sharedMap.threats = [];
multiAgentSystem.sharedMap.areasOfInterest = [];
multiAgentSystem.sharedMap.coverageGrid = zeros(100, 100);
multiAgentSystem.sharedMap.lastUpdate = datetime('now');
fprintf('Multi-agent system initialized with %d coordinated drones\n',
numDrones);
end
function templateDrone = createTemplateDrone()
% Create a template drone structure with all required fields
templateDrone = struct();
templateDrone.id = 0;
templateDrone.position = [0, 0, 0];
templateDrone.batteryLevel = 100;
templateDrone.status = 'inactive';
templateDrone.role = '';
templateDrone.capabilities = struct();
templateDrone.aiSystem = struct();
templateDrone.communicationBuffer = {};
templateDrone.taskQueue = {};
templateDrone.neighbors = [];
templateDrone.currentAction = '';
templateDrone.decisionConfidence = 0;
templateDrone.detections = struct('bboxes', [], 'scores', [], 'labels',
{{}}, 'confidence', 0, 'position', [0,0,0], 'timestamp', datetime('now'));
templateDrone.aiMetrics = struct('processingTime', 0, 'memoryUsed', 0,
'quantizationApplied', false);
templateDrone.receivedMessages = {};
end
function role = assignDroneRole(droneId, totalDrones)
roles = {'scout', 'surveillance', 'response'};
role = roles{mod(droneId-1, length(roles)) + 1};
end
function capabilities = defineDroneCapabilities(droneId)
capabilities = struct();
capabilities.maxSpeed = 15;
capabilities.sensorRange = 80;
capabilities.communicationRange = 100;
capabilities.maxFlightTime = 1800;
role = assignDroneRole(droneId, 3);
switch role
case 'scout'
capabilities.sensorPrecision = 'high';
capabilities.processorType = 'high_perf';
case 'surveillance'
capabilities.sensorPrecision = 'medium';
capabilities.processorType = 'efficient';
case 'response'
capabilities.sensorPrecision = 'low';

2

capabilities.processorType = 'balanced';
end
end
function aiSystem = initializeQuantizedDroneAI(role)
aiSystem = struct();
aiSystem.quantization = struct('enabled', true, 'precision', 'int8');
aiSystem.detector = createRoleSpecificDetector(role);
aiSystem.coordinationModel = loadCoordinationModel(role);
aiSystem.decisionHistory = [];
aiSystem.communicationEfficiency = 0.85;
end
function detector = createRoleSpecificDetector(role)
detector = struct();
switch role
case 'scout'
detector.inputSize = [128, 128, 3];
detector.quantizationBits = 8;
detector.classes = {'person', 'vehicle', 'suspicious_activity'};
case 'surveillance'
detector.inputSize = [128, 128, 3];
detector.quantizationBits = 8;
detector.classes = {'person', 'vehicle', 'animal', 'object'};
case 'response'
detector.inputSize = [128, 128, 3];
detector.quantizationBits = 16;
detector.classes = {'person', 'vehicle', 'threat'};
end
end
function model = loadCoordinationModel(role)
model = struct();
model.role = role;
model.taskWeights = defineTaskWeights(role);
model.communicationProtocol = defineCommunicationProtocol(role);
end
function weights = defineTaskWeights(role)
switch role
case 'scout'
weights = struct('exploration', 0.6, 'surveillance', 0.3,
'response', 0.1);
case 'surveillance'
weights = struct('exploration', 0.2, 'surveillance', 0.7,
'response', 0.1);
case 'response'
weights = struct('exploration', 0.1, 'surveillance', 0.3,
'response', 0.6);
end
end
function protocol = defineCommunicationProtocol(role)
protocol = struct();

3

switch role
case 'scout'
protocol.updateFrequency = 2;
protocol.dataPriority = 'high';
protocol.compression = 'lossy';
case 'surveillance'
protocol.updateFrequency = 1;
protocol.dataPriority = 'medium';
protocol.compression = 'balanced';
case 'response'
protocol.updateFrequency = 5;
protocol.dataPriority = 'critical';
protocol.compression = 'lossless';
end
end
% Coordinated Mission Execution
function multiAgentSystem = runCoordinatedMission(multiAgentSystem,
missionDuration)
fprintf('Running coordinated mission for %d seconds\n', missionDuration);
missionStart = tic;
timeStep = 0;
while toc(missionStart) < missionDuration &&
any([multiAgentSystem.drones.batteryLevel] > 5)
timeStep = timeStep + 1;
fprintf('\n--- Time Step %d ---\n', timeStep);
% Execute parallel drone operations
multiAgentSystem = executeDroneOperations(multiAgentSystem, timeStep);
% Multi-agent coordination and communication
multiAgentSystem = coordinateAgents(multiAgentSystem);
% Update shared situational awareness
multiAgentSystem = updateSharedAwareness(multiAgentSystem);
% Mission-level decision making
multiAgentSystem = makeMissionDecisions(multiAgentSystem);
% Display coordination status
if mod(timeStep, 3) == 0
displayCoordinationStatus(multiAgentSystem, timeStep);
end
pause(0.3); % Reduced pause for faster execution
end
fprintf('\nCoordinated mission completed!\n');
end
function multiAgentSystem = executeDroneOperations(multiAgentSystem, timeStep)
for i = 1:multiAgentSystem.numDrones

4

if strcmp(multiAgentSystem.drones(i).status, 'active') &&
multiAgentSystem.drones(i).batteryLevel > 5
% Update drone in place to maintain structure consistency
multiAgentSystem.drones(i) =
executeDroneCycle(multiAgentSystem.drones(i), timeStep);
end
end
end
function drone = executeDroneCycle(drone, timeStep)
% Simulate drone sensor data acquisition
sensorData = acquireSensorData(drone);
% Process with quantized AI
[detections, aiMetrics] = processWithQuantizedAI(drone.aiSystem,
sensorData);
% Update drone state (maintain structure consistency)
drone.detections = detections;
drone.aiMetrics = aiMetrics;
drone.batteryLevel = updateDroneBattery(drone, aiMetrics);
% Local decision making
drone = makeLocalDecision(drone, detections);
% Update communication buffer
drone = updateCommunicationBuffer(drone, detections, timeStep);
fprintf('Drone %d (%s): Battery=%.1f%%, Detections=%d, Action=%s\n', ...
drone.id, drone.role, drone.batteryLevel, length(detections.bboxes),
drone.currentAction);
end
function sensorData = acquireSensorData(drone)
sensorData = struct();
sensorData.timestamp = datetime('now');
sensorData.position = drone.position;
sensorData.image = generateSimulatedImage(drone);
sensorData.sensorReadings = generateSensorReadings(drone);
end
function image = generateSimulatedImage(drone)
% Use consistent image size for all roles
imageSize = [128, 128, 3];
image = uint8(rand(imageSize) * 255);
% Add simulated objects occasionally
if rand() > 0.5
image = addSimulatedObjects(image, drone.role);
end
end
function image = addSimulatedObjects(image, role)
[height, width, ~] = size(image);

5

switch role
case 'scout'
numObjects = randi([1, 2]);
objectTypes = {'person', 'vehicle', 'suspicious_activity'};
case 'surveillance'
numObjects = randi([0, 2]);
objectTypes = {'person', 'vehicle', 'animal', 'object'};
case 'response'
numObjects = randi([0, 1]);
objectTypes = {'person', 'vehicle', 'threat'};
end
for i = 1:numObjects
objectType = objectTypes{randi([1, length(objectTypes)])};
image = drawSimulatedObject(image, objectType);
end
end
function image = drawSimulatedObject(image, objectType)
[height, width, ~] = size(image);
switch objectType
case 'person'
color = [255, 0, 0];
sizeRange = [10, 20];
case 'vehicle'
color = [0, 0, 255];
sizeRange = [20, 40];
case 'threat'
color = [255, 255, 0];
sizeRange = [15, 30];
case 'animal'
color = [0, 255, 0];
sizeRange = [8, 15];
otherwise
color = [128, 128, 128];
sizeRange = [10, 25];
end
x = randi([1, width-50]);
y = randi([1, height-50]);
w = randi(sizeRange);
h = randi(sizeRange);
for c = 1:3
image(y:y+h, x:x+w, c) = color(c);
end
end
function readings = generateSensorReadings(drone)
readings = struct();
readings.temperature = 20 + randn() * 5;
readings.humidity = 50 + randn() * 10;

6

readings.windSpeed = 2 + rand() * 8;
readings.signalStrength = 80 + rand() * 20;
end
function [detections, aiMetrics] = processWithQuantizedAI(aiSystem,
sensorData)
tic;
if aiSystem.quantization.enabled
[output, metrics] = runQuantizedInference(aiSystem.detector,
sensorData.image);
else
[output, metrics] = runFullPrecisionInference(aiSystem.detector,
sensorData.image);
end
detections = convertToDetections(output, aiSystem.detector.classes);
detections.position = sensorData.position;
detections.timestamp = sensorData.timestamp;
aiMetrics.processingTime = toc;
aiMetrics.memoryUsed = metrics.memoryUsed;
aiMetrics.quantizationApplied = aiSystem.quantization.enabled;
end
function [output, metrics] = runQuantizedInference(detector, image)
metrics = struct();
% Simulate quantized inference
output = rand(1, length(detector.classes));
if detector.quantizationBits == 8
levels = 256;
output = round(output * levels) / levels;
end
output = abs(output);
if sum(output) > 0
output = output / sum(output);
else
output = ones(1, length(detector.classes)) / length(detector.classes);
end
metrics.memoryUsed = calculateQuantizedMemory(detector);
end
function [output, metrics] = runFullPrecisionInference(detector, image)
metrics = struct();
output = rand(1, length(detector.classes));
output = output / sum(output);
metrics.memoryUsed = calculateFullPrecisionMemory(detector);
end
function memory = calculateQuantizedMemory(detector)
bitsPerParam = detector.quantizationBits;

7

totalParams = 50000;
memory = (totalParams * bitsPerParam) / (8 * 1024);
end
function memory = calculateFullPrecisionMemory(detector)
bitsPerParam = 32;
totalParams = 50000;
memory = (totalParams * bitsPerParam) / (8 * 1024);
end
function detections = convertToDetections(output, classNames)
[maxScore, maxIdx] = max(output);
if maxScore > 0.3
detections.bboxes = [50, 50, 100, 100];
detections.scores = maxScore;
detections.labels = {classNames{maxIdx}};
detections.confidence = maxScore;
detections.position = [0, 0, 0];
detections.timestamp = datetime('now');
else
detections.bboxes = [];
detections.scores = [];
detections.labels = {};
detections.confidence = 0;
detections.position = [0, 0, 0];
detections.timestamp = datetime('now');
end
end
function batteryLevel = updateDroneBattery(drone, aiMetrics)
baseConsumption = 0.1; % Reduced consumption
% Adjust consumption based on quantization
if drone.aiSystem.quantization.enabled
energyMultiplier = 0.6;
else
energyMultiplier = 1.0;
end
% Role-specific consumption
switch drone.role
case 'scout'
roleMultiplier = 1.2;
case 'surveillance'
roleMultiplier = 1.0;
case 'response'
roleMultiplier = 1.3;
otherwise
roleMultiplier = 1.0;
end
consumption = baseConsumption * energyMultiplier * roleMultiplier;
batteryLevel = max(0, drone.batteryLevel - consumption);

8

end
function drone = makeLocalDecision(drone, detections)
% Local decision making based on role and current state
threatLevel = calculateThreatLevel(detections);
if drone.batteryLevel < 20
drone.currentAction = 'return_to_base';
elseif threatLevel > 0.7 && strcmp(drone.role, 'response')
drone.currentAction = 'engage_threat';
elseif threatLevel > 0.4
drone.currentAction = 'investigate';
else
drone.currentAction = 'patrol';
end
drone.decisionConfidence = threatLevel;
end
function threatLevel = calculateThreatLevel(detections)
threatLevel = 0;
if isempty(detections.labels)
return;
end
threatWeights = struct('person', 0.3, 'vehicle', 0.2, 'threat', 0.9, ...
'suspicious_activity', 0.8, 'animal', 0.1,
'object', 0.05);
for i = 1:length(detections.labels)
label = detections.labels{i};
if isfield(threatWeights, label)
threatLevel = threatLevel + threatWeights.(label) *
detections.scores(i);
end
end
threatLevel = min(1, threatLevel);
end
function drone = updateCommunicationBuffer(drone, detections, timeStep)
message = struct();
message.sender = drone.id;
message.timestamp = datetime('now');
message.position = drone.position;
message.detections = detections;
message.battery = drone.batteryLevel;
message.action = drone.currentAction;
% Apply communication compression based on role
message = compressMessage(message,
drone.aiSystem.coordinationModel.communicationProtocol);
drone.communicationBuffer{end+1} = message;

9

% Keep buffer size limited
if length(drone.communicationBuffer) > 5
drone.communicationBuffer = drone.communicationBuffer(end-4:end);
end
end
function message = compressMessage(message, protocol)
switch protocol.compression
case 'lossy'
message.position = round(message.position);
message.battery = round(message.battery * 10) / 10;
case 'balanced'
message.position = round(message.position / 5) * 5;
message.battery = round(message.battery * 5) / 5;
case 'lossless'
% No compression
end
end
% Multi-Agent Coordination
function multiAgentSystem = coordinateAgents(multiAgentSystem)
% Update neighbor relationships based on positions
multiAgentSystem = updateNeighborRelationships(multiAgentSystem);
% Exchange information between neighbors
multiAgentSystem = exchangeInformation(multiAgentSystem);
% Coordinate tasks and resolve conflicts
multiAgentSystem = coordinateTasks(multiAgentSystem);
end
function multiAgentSystem = updateNeighborRelationships(multiAgentSystem)
for i = 1:multiAgentSystem.numDrones
neighbors = [];
for j = 1:multiAgentSystem.numDrones
if i ~= j
distance = norm(multiAgentSystem.drones(i).position -
multiAgentSystem.drones(j).position);
if distance <= multiAgentSystem.communicationRange
neighbors(end+1) = j;
end
end
end
multiAgentSystem.drones(i).neighbors = neighbors;
end
end
function multiAgentSystem = exchangeInformation(multiAgentSystem)
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);

10

for j = drone.neighbors
if j > i % Avoid duplicate exchanges
multiAgentSystem = exchangeMessages(multiAgentSystem, i, j);
end
end
end
end
function multiAgentSystem = exchangeMessages(multiAgentSystem, drone1, drone2)
% Simulate message exchange between two drones
message1 = multiAgentSystem.drones(drone1).communicationBuffer;
message2 = multiAgentSystem.drones(drone2).communicationBuffer;
if ~isempty(message1)
% Drone2 receives latest message from Drone1
if isempty(multiAgentSystem.drones(drone2).receivedMessages)
multiAgentSystem.drones(drone2).receivedMessages = {};
end
multiAgentSystem.drones(drone2).receivedMessages{end+1} =
message1{end};
end
if ~isempty(message2)
% Drone1 receives latest message from Drone2
if isempty(multiAgentSystem.drones(drone1).receivedMessages)
multiAgentSystem.drones(drone1).receivedMessages = {};
end
multiAgentSystem.drones(drone1).receivedMessages{end+1} =
message2{end};
end
end
function multiAgentSystem = coordinateTasks(multiAgentSystem)
% Simple task coordination based on roles and proximity
for i = 1:multiAgentSystem.numDrones
if ~isempty(multiAgentSystem.drones(i).receivedMessages)
multiAgentSystem.drones(i) =
adjustBehaviorBasedOnMessages(multiAgentSystem.drones(i));
end
end
end
function drone = adjustBehaviorBasedOnMessages(drone)
for i = 1:length(drone.receivedMessages)
message = drone.receivedMessages{i};
% Adjust behavior based on neighbor information
if message.battery < 30 && drone.batteryLevel > 50
if strcmp(drone.currentAction, 'patrol')
drone.currentAction = 'assist_neighbor';
end
end
if message.detections.confidence > 0.7 && strcmp(drone.role,

11

'response')
if norm(drone.position - message.position) > 50
drone.currentAction = 'investigate_shared';
end
end
end
% Clear processed messages
drone.receivedMessages = {};
end
function multiAgentSystem = updateSharedAwareness(multiAgentSystem)
% Update shared map with information from all drones
allThreats = [];
coverageUpdate = zeros(100, 100);
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);
if ~isempty(drone.detections.bboxes) && drone.detections.confidence >
0.5
% Add detection to shared threats
threat = struct();
threat.position = drone.position;
threat.confidence = drone.detections.confidence;
threat.type = drone.detections.labels;
threat.source = drone.id;
threat.timestamp = datetime('now');
allThreats = [allThreats, threat];
end
% Update coverage grid
gridX = max(1, min(100, round(drone.position(1) / 5)));
gridY = max(1, min(100, round(drone.position(2) / 5)));
coverageUpdate(gridY, gridX) = coverageUpdate(gridY, gridX) + 1;
end
multiAgentSystem.sharedMap.threats = allThreats;
multiAgentSystem.sharedMap.coverageGrid =
multiAgentSystem.sharedMap.coverageGrid + coverageUpdate;
multiAgentSystem.sharedMap.lastUpdate = datetime('now');
end
function multiAgentSystem = makeMissionDecisions(multiAgentSystem)
% Mission-level decision making based on shared awareness
% Check if any area needs reinforcement
coverage = multiAgentSystem.sharedMap.coverageGrid;
[minCoverage, minIdx] = min(coverage(:));
if minCoverage == 0
[y, x] = ind2sub(size(coverage), minIdx);
targetPos = [x*5, y*5, 50];

12

% Assign closest available drone to cover the area
availableDrones = find(strcmp({multiAgentSystem.drones.status},
'active'));
if ~isempty(availableDrones)
distances = arrayfun(@(i)
norm(multiAgentSystem.drones(i).position - targetPos), availableDrones);
[~, closestIdx] = min(distances);
assignedDrone = availableDrones(closestIdx);
multiAgentSystem.drones(assignedDrone).currentAction =
'cover_gap';
end
end
end
function displayCoordinationStatus(multiAgentSystem, timeStep)
fprintf('\n=== Coordination Status (Time: %d) ===\n', timeStep);
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);
fprintf('Drone %d (%s): Battery=%.1f%%, Action=%s, Neighbors=%d\n',
...
drone.id, drone.role, drone.batteryLevel, drone.currentAction,
length(drone.neighbors));
end
coveragePercentage = nnz(multiAgentSystem.sharedMap.coverageGrid) /
numel(multiAgentSystem.sharedMap.coverageGrid) * 100;
fprintf('Shared Threats: %d, Coverage: %.1f%%\n',
length(multiAgentSystem.sharedMap.threats), coveragePercentage);
end
function generateMultiAgentReport(multiAgentSystem)
fprintf('\n=== MULTI-AGENT QUANTIZED SURVEILLANCE REPORT ===\n');
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);
fprintf('\nDrone %d (%s):\n', drone.id, drone.role);
fprintf(' Battery: %.1f%%, Position: [%.1f, %.1f, %.1f]\n', ...
drone.batteryLevel, drone.position(1), drone.position(2),
drone.position(3));
fprintf(' Status: %s, Quantization: %s\n', drone.status,
string(drone.aiSystem.quantization.enabled));
end
% System-wide metrics
coveragePercentage = nnz(multiAgentSystem.sharedMap.coverageGrid) /
numel(multiAgentSystem.sharedMap.coverageGrid) * 100;
fprintf('\n=== SYSTEM PERFORMANCE ===\n');
fprintf('Total Drones: %d\n', multiAgentSystem.numDrones);
fprintf('Area Coverage: %.1f%%\n', coveragePercentage);
fprintf('Total Threats Detected: %d\n',

13

length(multiAgentSystem.sharedMap.threats));
fprintf('Coordination Mode: %s\n', multiAgentSystem.coordinationMode);
% Plot results
plotMultiAgentResults(multiAgentSystem);
end
function plotMultiAgentResults(multiAgentSystem)
figure('Position', [100, 100, 1200, 800]);
% Drone positions and coverage
subplot(2,3,1);
hold on;
colors = ['r', 'g', 'b', 'c', 'm', 'y'];
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);
plot(drone.position(1), drone.position(2), [colors(i) 'o'],
'MarkerSize', 10, 'LineWidth', 2);
text(drone.position(1)+5, drone.position(2)+5, sprintf('D%d (%s)',
drone.id, drone.role));
end
coverage = multiAgentSystem.sharedMap.coverageGrid;
imagesc([0, 500], [0, 500], coverage);
colorbar;
title('Drone Positions & Coverage');
xlabel('X Position');
ylabel('Y Position');
grid on;
% Battery levels
subplot(2,3,2);
batteryLevels = [multiAgentSystem.drones.batteryLevel];
bar(batteryLevels, 'FaceColor', [0.2, 0.6, 0.8]);
title('Final Battery Levels');
xlabel('Drone ID');
ylabel('Battery Level (%)');
set(gca, 'XTickLabel', 1:multiAgentSystem.numDrones);
grid on;
% Role distribution
subplot(2,3,3);
roles = {multiAgentSystem.drones.role};
[uniqueRoles, ~, roleIdx] = unique(roles);
roleCounts = histcounts(roleIdx, 1:length(uniqueRoles)+1);
pie(roleCounts, uniqueRoles);
title('Drone Role Distribution');
% Communication network
subplot(2,3,4);
hold on;
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);

14

for j = drone.neighbors
if j > i
pos1 = multiAgentSystem.drones(i).position;
pos2 = multiAgentSystem.drones(j).position;
plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'k-',
'LineWidth', 1);
end
end
end
for i = 1:multiAgentSystem.numDrones
drone = multiAgentSystem.drones(i);
plot(drone.position(1), drone.position(2), [colors(i) 'o'],
'MarkerSize', 8, 'LineWidth', 2);
end
title('Communication Network');
xlabel('X Position');
ylabel('Y Position');
grid on;
% Quantization benefits
subplot(2,3,5);
methods = {'FP32', 'INT8', 'INT16'};
memoryUsage = [1.0, 0.25, 0.5];
energyUsage = [1.0, 0.4, 0.6];
bar([memoryUsage; energyUsage]');
legend('Memory', 'Energy', 'Location', 'northwest');
title('Quantization Benefits');
set(gca, 'XTickLabel', methods);
ylabel('Relative Usage');
grid on;
% Multi-agent efficiency
subplot(2,3,6);
droneCount = 1:multiAgentSystem.numDrones;
coverageGain = [30, 60, 85, 95]; % Example coverage gains
plot(1:length(coverageGain), coverageGain, 'b-', 'LineWidth', 2);
title('Coverage vs Drone Count');
xlabel('Number of Drones');
ylabel('Area Coverage (%)');
grid on;
sgtitle('Multi-Agent Quantized Surveillance Analysis', 'FontSize', 14,
'FontWeight', 'bold');
end
