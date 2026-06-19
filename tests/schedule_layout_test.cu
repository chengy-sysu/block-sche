#include <block_sche/block_sche.cuh>

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  auto schedule = block_sche::make_schedule(block_sche::Dim3u(8, 1, 1), 4, 2);
  assert(schedule.logical_grid().x == 8);
  assert(schedule.resident_ctas() == 8);
  assert(schedule.sm_count() == 4);
  assert(schedule.task_count() == 8);

  const auto& offsets = schedule.sm_offsets();
  const auto& ids = schedule.sm_task_ids();
  assert(offsets.size() == 5);
  assert(offsets[0] == 0);
  assert(offsets[1] == 2);
  assert(offsets[2] == 4);
  assert(offsets[3] == 6);
  assert(offsets[4] == 8);

  assert(schedule.tasks()[ids[offsets[0]]].linear_block == 0);
  assert(schedule.tasks()[ids[offsets[0] + 1]].linear_block == 4);
  assert(schedule.tasks()[ids[offsets[1]]].linear_block == 1);
  assert(schedule.tasks()[ids[offsets[1] + 1]].linear_block == 5);

  auto blocked = block_sche::make_schedule(block_sche::Dim3u(8, 1, 1), 4, 1,
                                           block_sche::IdentityBlockOrder(),
                                           block_sche::BlockedSM(2));
  assert(blocked.sm_offsets()[1] == 2);
  assert(blocked.sm_offsets()[2] == 4);
  assert(blocked.tasks()[blocked.sm_task_ids()[0]].sm == 0);
  assert(blocked.tasks()[blocked.sm_task_ids()[1]].sm == 0);
  assert(blocked.tasks()[blocked.sm_task_ids()[2]].sm == 1);
  assert(blocked.tasks()[blocked.sm_task_ids()[3]].sm == 1);

  auto column_major = block_sche::ScheduleBuilder(block_sche::Dim3u(2, 3, 1))
                          .sms(2)
                          .ctas_per_sm(1)
                          .column_major_round_robin();
  assert(column_major.tasks()[0].linear_block == 0);
  assert(column_major.tasks()[1].linear_block == 2);
  assert(column_major.tasks()[2].linear_block == 4);
  assert(column_major.tasks()[3].linear_block == 1);
  assert(column_major.tasks()[4].linear_block == 3);
  assert(column_major.tasks()[5].linear_block == 5);

  std::vector<uint32_t> explicit_map = {1, 1, 0, 0, 1, 0};
  auto explicit_schedule = block_sche::ScheduleBuilder(block_sche::Dim3u(6, 1, 1))
                               .sms(2)
                               .ctas_per_sm(2)
                               .explicit_sm(explicit_map);
  assert(explicit_schedule.resident_ctas() == 4);
  assert(explicit_schedule.sm_offsets()[0] == 0);
  assert(explicit_schedule.sm_offsets()[1] == 3);
  assert(explicit_schedule.sm_offsets()[2] == 6);
  assert(explicit_schedule.tasks()[0].sm == 1);
  assert(explicit_schedule.tasks()[2].sm == 0);

  block_sche::PreparedSchedule prepared;
  assert(prepared.task_count() == 0);
  assert(prepared.host().task_count() == 0);

  block_sche::PreparedTrace prepared_trace;
  assert(prepared_trace.size() == 0);
  return 0;
}
