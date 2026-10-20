import numpy as np

from safety.perception import check_classes, results_to_dets


class FakeTensor:
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = FakeTensor(xyxy), FakeTensor(conf), FakeTensor(cls)

    def __len__(self):
        return len(self.cls.a)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


NAMES = ["helmet", "no-helmet", "person"]


def test_results_are_converted_to_dets():
    res = FakeResult(FakeBoxes([[10, 20, 50, 90], [0, 0, 5, 5]], [0.9, 0.4], [2, 0]))
    dets = results_to_dets(res, NAMES)
    assert [d.cls for d in dets] == ["person", "helmet"]
    assert dets[0].conf == np.float32(0.9) or abs(dets[0].conf - 0.9) < 1e-6
    assert dets[0].box == (10.0, 20.0, 50.0, 90.0)


def test_no_detections_gives_empty_list():
    assert results_to_dets(FakeResult(None), NAMES) == []
    assert results_to_dets(FakeResult(FakeBoxes(np.zeros((0, 4)), [], [])), NAMES) == []


def test_check_classes_reports_missing_names():
    assert check_classes(NAMES, ["person", "helmet"]) == []
    assert check_classes(NAMES, ["person", "Person", "vest"]) == ["Person", "vest"]
