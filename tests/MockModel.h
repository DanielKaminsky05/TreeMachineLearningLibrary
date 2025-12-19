#ifndef MOCKMODEL_H
#define MOCKMODEL_H

#include "gmock/gmock.h"
#include "../code/MLSuite/IModel.h"

class MockModel : public IModel {
public:
    MOCK_METHOD(void, fit, (const std::vector<float>&, const std::vector<std::string>&, const std::vector<float>&), (override));
    MOCK_METHOD(std::vector<float>, predict, (const std::vector<float>&, const std::vector<std::string>&), (const, override));
    MOCK_METHOD(std::string, getName, (), (const, override));
};

#endif // MOCKMODEL_H
