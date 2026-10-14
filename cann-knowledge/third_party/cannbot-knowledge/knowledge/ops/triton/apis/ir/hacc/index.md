# HACC 方言

HACC 方言建模 Host-Device 异构调用边界：函数角色、参数语义、设备规格与 Host 辅助函数绑定。

- [overview](overview.md)：方言定位、四大核心概念与典型 IR 形态
- [function_management](function_management.md)：HACCFuncType（HOST/DEVICE）、HACCFunctionInterface 与辅助属性
- [device_specification](device_specification.md)：DeviceSpecEnum 14 字段、TargetDeviceSpecAttr 与支持的 NPU 型号
- [kernel_args](kernel_args.md)：KernelArgType 12 类参数语义、out-param 约定与 tiling 参数打包
- [host_device_binding](host_device_binding.md)：HostFuncType 8 类 Host 辅助函数角色与 FuncRefAttr 符号绑定
- [transforms](transforms.md)：rename-func 与 append-device-spec 两个变换 Pass
