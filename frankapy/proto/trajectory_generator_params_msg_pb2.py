# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trajectory_generator_params_msg.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='trajectory_generator_params_msg.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n%trajectory_generator_params_msg.proto\"_\n!GripperTrajectoryGeneratorMessage\x12\r\n\x05grasp\x18\x01 \x02(\x08\x12\r\n\x05width\x18\x02 \x02(\x01\x12\r\n\x05speed\x18\x03 \x02(\x01\x12\r\n\x05\x66orce\x18\x04 \x02(\x01\"\x8c\x01\n!ImpulseTrajectoryGeneratorMessage\x12\x10\n\x08run_time\x18\x01 \x02(\x01\x12\x10\n\x08\x61\x63\x63_time\x18\x02 \x02(\x01\x12\x11\n\tmax_trans\x18\x03 \x02(\x01\x12\x0f\n\x07max_rot\x18\x04 \x02(\x01\x12\x0e\n\x06\x66orces\x18\x05 \x03(\x01\x12\x0f\n\x07torques\x18\x06 \x03(\x01\"C\n\x1fJointTrajectoryGeneratorMessage\x12\x10\n\x08run_time\x18\x01 \x02(\x01\x12\x0e\n\x06joints\x18\x02 \x03(\x01\"f\n\x1ePoseTrajectoryGeneratorMessage\x12\x10\n\x08run_time\x18\x01 \x02(\x01\x12\x10\n\x08position\x18\x02 \x03(\x01\x12\x12\n\nquaternion\x18\x03 \x03(\x01\x12\x0c\n\x04pose\x18\x04 \x03(\x01\"\xe5\x01\n\"JointDMPTrajectoryGeneratorMessage\x12\x10\n\x08run_time\x18\x01 \x02(\x01\x12\x0b\n\x03tau\x18\x02 \x02(\x01\x12\r\n\x05\x61lpha\x18\x03 \x02(\x01\x12\x0c\n\x04\x62\x65ta\x18\x04 \x02(\x01\x12\x11\n\tnum_basis\x18\x05 \x02(\x01\x12\x19\n\x11num_sensor_values\x18\x06 \x02(\x01\x12\x12\n\nbasis_mean\x18\x07 \x03(\x01\x12\x11\n\tbasis_std\x18\x08 \x03(\x01\x12\x0f\n\x07weights\x18\t \x03(\x01\x12\x1d\n\x15initial_sensor_values\x18\n \x03(\x01\"\xa7\x02\n!PoseDMPTrajectoryGeneratorMessage\x12\x18\n\x10orientation_only\x18\x01 \x02(\x08\x12\x15\n\rposition_only\x18\x02 \x02(\x08\x12\x10\n\x08\x65\x65_frame\x18\x03 \x02(\x08\x12\x10\n\x08run_time\x18\x04 \x02(\x01\x12\x0b\n\x03tau\x18\x05 \x02(\x01\x12\r\n\x05\x61lpha\x18\x06 \x02(\x01\x12\x0c\n\x04\x62\x65ta\x18\x07 \x02(\x01\x12\x11\n\tnum_basis\x18\x08 \x02(\x01\x12\x19\n\x11num_sensor_values\x18\t \x02(\x01\x12\x12\n\nbasis_mean\x18\n \x03(\x01\x12\x11\n\tbasis_std\x18\x0b \x03(\x01\x12\x0f\n\x07weights\x18\x0c \x03(\x01\x12\x1d\n\x15initial_sensor_values\x18\r \x03(\x01\"\xec\x04\n+QuaternionPoseDMPTrajectoryGeneratorMessage\x12\x10\n\x08\x65\x65_frame\x18\x01 \x02(\x08\x12\x10\n\x08run_time\x18\x02 \x02(\x01\x12\x0f\n\x07tau_pos\x18\x03 \x02(\x01\x12\x11\n\talpha_pos\x18\x04 \x02(\x01\x12\x10\n\x08\x62\x65ta_pos\x18\x05 \x02(\x01\x12\x10\n\x08tau_quat\x18\x06 \x02(\x01\x12\x12\n\nalpha_quat\x18\x07 \x02(\x01\x12\x11\n\tbeta_quat\x18\x08 \x02(\x01\x12\x15\n\rnum_basis_pos\x18\t \x02(\x01\x12\x1d\n\x15num_sensor_values_pos\x18\n \x02(\x01\x12\x16\n\x0enum_basis_quat\x18\x0b \x02(\x01\x12\x1e\n\x16num_sensor_values_quat\x18\x0c \x02(\x01\x12\x16\n\x0epos_basis_mean\x18\r \x03(\x01\x12\x15\n\rpos_basis_std\x18\x0e \x03(\x01\x12\x13\n\x0bpos_weights\x18\x0f \x03(\x01\x12!\n\x19pos_initial_sensor_values\x18\x10 \x03(\x01\x12\x17\n\x0fquat_basis_mean\x18\x11 \x03(\x01\x12\x16\n\x0equat_basis_std\x18\x12 \x03(\x01\x12\x14\n\x0cquat_weights\x18\x13 \x03(\x01\x12\"\n\x1aquat_initial_sensor_values\x18\x14 \x03(\x01\x12\x16\n\x0egoal_time_quat\x18\x15 \x02(\x01\x12\x13\n\x0bgoal_quat_w\x18\x16 \x02(\x01\x12\x13\n\x0bgoal_quat_x\x18\x17 \x02(\x01\x12\x13\n\x0bgoal_quat_y\x18\x18 \x02(\x01\x12\x13\n\x0bgoal_quat_z\x18\x19 \x02(\x01\"\"\n\x0eRunTimeMessage\x12\x10\n\x08run_time\x18\x01 \x02(\x01'
)




_GRIPPERTRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='GripperTrajectoryGeneratorMessage',
  full_name='GripperTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='grasp', full_name='GripperTrajectoryGeneratorMessage.grasp', index=0,
      number=1, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='GripperTrajectoryGeneratorMessage.width', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speed', full_name='GripperTrajectoryGeneratorMessage.speed', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force', full_name='GripperTrajectoryGeneratorMessage.force', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=41,
  serialized_end=136,
)


_IMPULSETRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='ImpulseTrajectoryGeneratorMessage',
  full_name='ImpulseTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='run_time', full_name='ImpulseTrajectoryGeneratorMessage.run_time', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='acc_time', full_name='ImpulseTrajectoryGeneratorMessage.acc_time', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_trans', full_name='ImpulseTrajectoryGeneratorMessage.max_trans', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_rot', full_name='ImpulseTrajectoryGeneratorMessage.max_rot', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='forces', full_name='ImpulseTrajectoryGeneratorMessage.forces', index=4,
      number=5, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='torques', full_name='ImpulseTrajectoryGeneratorMessage.torques', index=5,
      number=6, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=139,
  serialized_end=279,
)


_JOINTTRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='JointTrajectoryGeneratorMessage',
  full_name='JointTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='run_time', full_name='JointTrajectoryGeneratorMessage.run_time', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='joints', full_name='JointTrajectoryGeneratorMessage.joints', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=281,
  serialized_end=348,
)


_POSETRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='PoseTrajectoryGeneratorMessage',
  full_name='PoseTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='run_time', full_name='PoseTrajectoryGeneratorMessage.run_time', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='position', full_name='PoseTrajectoryGeneratorMessage.position', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quaternion', full_name='PoseTrajectoryGeneratorMessage.quaternion', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pose', full_name='PoseTrajectoryGeneratorMessage.pose', index=3,
      number=4, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=350,
  serialized_end=452,
)


_JOINTDMPTRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='JointDMPTrajectoryGeneratorMessage',
  full_name='JointDMPTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='run_time', full_name='JointDMPTrajectoryGeneratorMessage.run_time', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tau', full_name='JointDMPTrajectoryGeneratorMessage.tau', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='JointDMPTrajectoryGeneratorMessage.alpha', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta', full_name='JointDMPTrajectoryGeneratorMessage.beta', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_basis', full_name='JointDMPTrajectoryGeneratorMessage.num_basis', index=4,
      number=5, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_sensor_values', full_name='JointDMPTrajectoryGeneratorMessage.num_sensor_values', index=5,
      number=6, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basis_mean', full_name='JointDMPTrajectoryGeneratorMessage.basis_mean', index=6,
      number=7, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basis_std', full_name='JointDMPTrajectoryGeneratorMessage.basis_std', index=7,
      number=8, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weights', full_name='JointDMPTrajectoryGeneratorMessage.weights', index=8,
      number=9, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initial_sensor_values', full_name='JointDMPTrajectoryGeneratorMessage.initial_sensor_values', index=9,
      number=10, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=455,
  serialized_end=684,
)


_POSEDMPTRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='PoseDMPTrajectoryGeneratorMessage',
  full_name='PoseDMPTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='orientation_only', full_name='PoseDMPTrajectoryGeneratorMessage.orientation_only', index=0,
      number=1, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='position_only', full_name='PoseDMPTrajectoryGeneratorMessage.position_only', index=1,
      number=2, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ee_frame', full_name='PoseDMPTrajectoryGeneratorMessage.ee_frame', index=2,
      number=3, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_time', full_name='PoseDMPTrajectoryGeneratorMessage.run_time', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tau', full_name='PoseDMPTrajectoryGeneratorMessage.tau', index=4,
      number=5, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='PoseDMPTrajectoryGeneratorMessage.alpha', index=5,
      number=6, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta', full_name='PoseDMPTrajectoryGeneratorMessage.beta', index=6,
      number=7, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_basis', full_name='PoseDMPTrajectoryGeneratorMessage.num_basis', index=7,
      number=8, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_sensor_values', full_name='PoseDMPTrajectoryGeneratorMessage.num_sensor_values', index=8,
      number=9, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basis_mean', full_name='PoseDMPTrajectoryGeneratorMessage.basis_mean', index=9,
      number=10, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basis_std', full_name='PoseDMPTrajectoryGeneratorMessage.basis_std', index=10,
      number=11, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weights', full_name='PoseDMPTrajectoryGeneratorMessage.weights', index=11,
      number=12, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initial_sensor_values', full_name='PoseDMPTrajectoryGeneratorMessage.initial_sensor_values', index=12,
      number=13, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=687,
  serialized_end=982,
)


_QUATERNIONPOSEDMPTRAJECTORYGENERATORMESSAGE = _descriptor.Descriptor(
  name='QuaternionPoseDMPTrajectoryGeneratorMessage',
  full_name='QuaternionPoseDMPTrajectoryGeneratorMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ee_frame', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.ee_frame', index=0,
      number=1, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_time', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.run_time', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tau_pos', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.tau_pos', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha_pos', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.alpha_pos', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta_pos', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.beta_pos', index=4,
      number=5, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tau_quat', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.tau_quat', index=5,
      number=6, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha_quat', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.alpha_quat', index=6,
      number=7, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta_quat', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.beta_quat', index=7,
      number=8, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_basis_pos', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.num_basis_pos', index=8,
      number=9, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_sensor_values_pos', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.num_sensor_values_pos', index=9,
      number=10, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_basis_quat', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.num_basis_quat', index=10,
      number=11, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_sensor_values_quat', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.num_sensor_values_quat', index=11,
      number=12, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_basis_mean', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.pos_basis_mean', index=12,
      number=13, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_basis_std', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.pos_basis_std', index=13,
      number=14, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_weights', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.pos_weights', index=14,
      number=15, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_initial_sensor_values', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.pos_initial_sensor_values', index=15,
      number=16, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quat_basis_mean', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.quat_basis_mean', index=16,
      number=17, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quat_basis_std', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.quat_basis_std', index=17,
      number=18, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quat_weights', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.quat_weights', index=18,
      number=19, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quat_initial_sensor_values', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.quat_initial_sensor_values', index=19,
      number=20, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal_time_quat', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.goal_time_quat', index=20,
      number=21, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal_quat_w', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.goal_quat_w', index=21,
      number=22, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal_quat_x', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.goal_quat_x', index=22,
      number=23, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal_quat_y', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.goal_quat_y', index=23,
      number=24, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal_quat_z', full_name='QuaternionPoseDMPTrajectoryGeneratorMessage.goal_quat_z', index=24,
      number=25, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=985,
  serialized_end=1605,
)


_RUNTIMEMESSAGE = _descriptor.Descriptor(
  name='RunTimeMessage',
  full_name='RunTimeMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='run_time', full_name='RunTimeMessage.run_time', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1607,
  serialized_end=1641,
)

DESCRIPTOR.message_types_by_name['GripperTrajectoryGeneratorMessage'] = _GRIPPERTRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['ImpulseTrajectoryGeneratorMessage'] = _IMPULSETRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['JointTrajectoryGeneratorMessage'] = _JOINTTRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['PoseTrajectoryGeneratorMessage'] = _POSETRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['JointDMPTrajectoryGeneratorMessage'] = _JOINTDMPTRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['PoseDMPTrajectoryGeneratorMessage'] = _POSEDMPTRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['QuaternionPoseDMPTrajectoryGeneratorMessage'] = _QUATERNIONPOSEDMPTRAJECTORYGENERATORMESSAGE
DESCRIPTOR.message_types_by_name['RunTimeMessage'] = _RUNTIMEMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GripperTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('GripperTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _GRIPPERTRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:GripperTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(GripperTrajectoryGeneratorMessage)

ImpulseTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('ImpulseTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _IMPULSETRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:ImpulseTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(ImpulseTrajectoryGeneratorMessage)

JointTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('JointTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _JOINTTRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:JointTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(JointTrajectoryGeneratorMessage)

PoseTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('PoseTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _POSETRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:PoseTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(PoseTrajectoryGeneratorMessage)

JointDMPTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('JointDMPTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _JOINTDMPTRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:JointDMPTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(JointDMPTrajectoryGeneratorMessage)

PoseDMPTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('PoseDMPTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _POSEDMPTRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:PoseDMPTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(PoseDMPTrajectoryGeneratorMessage)

QuaternionPoseDMPTrajectoryGeneratorMessage = _reflection.GeneratedProtocolMessageType('QuaternionPoseDMPTrajectoryGeneratorMessage', (_message.Message,), {
  'DESCRIPTOR' : _QUATERNIONPOSEDMPTRAJECTORYGENERATORMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:QuaternionPoseDMPTrajectoryGeneratorMessage)
  })
_sym_db.RegisterMessage(QuaternionPoseDMPTrajectoryGeneratorMessage)

RunTimeMessage = _reflection.GeneratedProtocolMessageType('RunTimeMessage', (_message.Message,), {
  'DESCRIPTOR' : _RUNTIMEMESSAGE,
  '__module__' : 'trajectory_generator_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:RunTimeMessage)
  })
_sym_db.RegisterMessage(RunTimeMessage)


# @@protoc_insertion_point(module_scope)
