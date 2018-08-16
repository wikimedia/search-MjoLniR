import kafka
from kafka.structs import OffsetAndTimestamp

from mjolnir.kafka import client


def test_offset_for_times(mocker):
    partitions = [kafka.TopicPartition('ut_topic', 0)]
    offsets_for_times = {tp: OffsetAndTimestamp(42, -1) for tp in partitions}
    positions = {tp: 747 for tp in partitions}

    mock = mocker.Mock()
    mock.offsets_for_times.return_value = offsets_for_times
    mock.position.side_effect = lambda tp: positions.get(tp, 0)

    # Uses returned offset for time when provided
    offsets = client.offsets_for_times(mock, partitions, 987654321)
    assert len(offsets) == len(partitions)
    assert all(tp in offsets for tp in partitions)
    assert offsets[partitions[0]] == 42

    # When offsets_for_times returns None returns position at end
    offsets_for_times[partitions[0]] = None
    offsets = client.offsets_for_times(mock, partitions, 987654321)
    assert len(offsets) == len(partitions)
    assert all(tp in offsets for tp in partitions)
    assert offsets[partitions[0]] == 747
