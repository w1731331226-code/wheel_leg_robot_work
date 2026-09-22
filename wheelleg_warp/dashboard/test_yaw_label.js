const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const lines=fs.readFileSync(path.join(__dirname,'app.js'),'utf8').split('\n');
const format=vm.runInNewContext(lines.find(x=>x.startsWith('function yawLabel('))+';yawLabel');
for(const value of [null,undefined,NaN,Infinity])assert.equal(format(value),'Jψ —（轨迹未完整）');
assert.equal(format(0),'Jψ 0.000°');assert.equal(format(1.23456),'Jψ 1.235°');
const labels=vm.runInNewContext(lines[0]+';terrainNames');
assert.equal(labels.single_side_ramp,'单侧坡道');assert.equal(labels.asymmetric_rough,'左右独立粗糙路');
console.log('PASS: incomplete yaw metrics and advanced terrain labels');
