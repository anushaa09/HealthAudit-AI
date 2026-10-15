// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract DecisionRegistry {
    struct Record {
        string hash;
        uint256 timestamp;
    }

    mapping(string => Record) private records;

    event DecisionAnchored(string indexed decisionId, string hash, uint256 timestamp);

    function storeHash(string memory decisionId, string memory hash) public {
        records[decisionId] = Record(hash, block.timestamp);
        emit DecisionAnchored(decisionId, hash, block.timestamp);
    }

    function getHash(string memory decisionId) public view returns (string memory, uint256) {
        Record memory r = records[decisionId];
        return (r.hash, r.timestamp);
    }
}